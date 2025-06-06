from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from .generator import WALGenerator

log = get_pylogger(__name__)

class WALEnv(RL4COEnvBase):
    """Walmart Location Problem (WAL) environment
    At each step, the agent chooses a location. The reward is 0 unless enough number of locations are chosen.
    The reward is (-) the total distance of each location to its closest chosen location.

    Observations:
        - the locations
        - the number of locations to choose

    Constraints:
        - the given number of locations must be chosen

    Finish condition:
        - the given number of locations are chosen

    Reward:
        - (minus) the total distance of each location to its closest chosen location

    Args:
        generator: WALGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "wal"

    def __init__(
        self,
        generator: WALGenerator = None,
        generator_params: dict = {},
        check_solution=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = WALGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
        self._make_spec(self.generator)
        
        # WAL-specific parameters
        self.lambda_g, self.lambda_f, self.tau, self.miu_nu_other = 1.938, 1.912, 3.5, 0.19

    def _step(self, td: TensorDict) -> TensorDict:
        selected = td["action"]
        batch_size = selected.shape[0]

        # Update location selection count
        chosen_count = td["chosen_count"].clone()  # (batch_size, n_locations)

        # add 1 to the selected index for each batch
        chosen_count[torch.arange(batch_size).to(td.device), selected] += 1

        # We are done if we choose enough locations
        done = td["i"] >= (td["to_choose"] - 1)

        reward = torch.zeros_like(done)

        # only mask when chosen_count >= 2
        action_mask = ~(chosen_count >= 2)

        td.update(
            {
                "chosen_count": chosen_count,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def reset(self, td: Optional[TensorDict] = None, batch_size=None, phase="train") -> TensorDict:
        """Reset function with device-aware generation to avoid CPU-GPU transfers"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        
        if td is None or td.is_empty():
            # Get device from environment if available
            device = getattr(self, 'device', torch.device('cpu'))
            # Generate data directly on target device
            td = self.generator._generate(batch_size, device=device, phase=phase)
            print(f"🚀 Generated TensorDict directly on {device}")
        
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        
        # Ensure environment is on the same device as data
        self.to(td.device)
        
        # Call our custom _reset method
        return self._reset(td, batch_size=batch_size)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # td should already be on the correct device at this point
        assert td is not None, "TensorDict should not be None in _reset"
        
        # Dynamically compute WAL features when resetting the environment
        # This allows us to compute features per batch instead of for the entire dataset
        wal_features = self._compute_wal_features_dynamic(td["locs"], batch_size)

        # Convert sparse tensors to dense only if needed for TensorDict cloning compatibility
        if td.device.type == 'mps':
            # On MPS, sparse tensors may cause cloning issues, so convert to dense
            wal_features = self._convert_sparse_to_dense_if_needed(wal_features)

        return TensorDict(
            {
                # given information
                "locs": td["locs"],  # (batch_size, n_points, dim_loc)
                # Dynamically computed WAL features (could be sparse or dense)
                **wal_features,
                # states changed by actions
                "chosen_count": torch.zeros(
                    *td["locs"].shape[:-1], dtype=torch.int64, device=td.device
                ),
                "to_choose": td["to_choose"],  # the number of sets to choose
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),  # the number of sets we have chosen
                "action_mask": torch.ones(
                    *td["locs"].shape[:-1], dtype=torch.bool, device=td.device
                ),
                # Initialize done state - required by constructive policy
                "done": torch.zeros(
                    *batch_size, dtype=torch.bool, device=td.device
                ),
            },
            batch_size=batch_size,
        )
    
    def _convert_sparse_to_dense_if_needed(self, features_dict):
        """Convert sparse tensors to dense only if they cause cloning issues"""
        converted = {}
        for key, value in features_dict.items():
            if hasattr(value, 'is_sparse') and value.is_sparse:
                converted[key] = value.to_dense()
            else:
                converted[key] = value
        return converted
    
    def _compute_wal_features_dynamic(self, locs, batch_size):
        """Dynamically compute WAL features for the current batch"""
        return self.generator.compute_wal_features_on_demand(locs, batch_size)

    def _make_spec(self, generator: WALGenerator):
        # TODO: make spec
        pass

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """
        Use precomputed WAL features to calculate reward
        """
        chosen_count = td["chosen_count"]
        device = chosen_count.device

        # Get features from td (could be sparse or dense tensor, depending on device compatibility)
        u_lj_reg_input = td["u_lj_reg"]  # [batch, N, block_num] - could be sparse or dense
        u_lj_food_input = td["u_lj_food"]  # [batch, N, block_num] - could be sparse or dense
        u0 = td["u0"]  # [batch, block_num]
        delivery_distances_reg_set = td["delivery_distances_reg_set"]  # [batch, N, 1]
        delivery_distances_food_set = td["delivery_distances_food_set"]  # [batch, N, 1]
        fixed_cost_point = td["fixed_cost_point"]  # [batch, 1, N]
        pop = td["pop"]  # [batch, block_num, 1]

        # Convert to dense if sparse for computation
        if hasattr(u_lj_reg_input, 'is_sparse') and u_lj_reg_input.is_sparse:
            u_lj_reg = u_lj_reg_input.to_dense()
        else:
            u_lj_reg = u_lj_reg_input
            
        if hasattr(u_lj_food_input, 'is_sparse') and u_lj_food_input.is_sparse:
            u_lj_food = u_lj_food_input.to_dense()
        else:
            u_lj_food = u_lj_food_input

        regular = (chosen_count >= 1).float()  # [batch, store_num]
        food = (chosen_count >= 2).float()    # [batch, store_num]

        # calculate regular stores
        u_sum_reg = torch.bmm(u_lj_reg.transpose(1, 2), regular.unsqueeze(-1)).squeeze(-1)
        P_sum_reg = u_sum_reg + u0 + 1e-8  # add numerical stability constant
        # use torch.clamp to ensure denominator is not too small
        P_sum_reg = torch.clamp(P_sum_reg, min=1e-6)
        revenue_g_block = torch.sum(pop.squeeze(-1) * (u_sum_reg / P_sum_reg) * self.lambda_g, dim=1)  # [batch]
        delivery_cost_reg = self.tau * torch.sum(regular * delivery_distances_reg_set.squeeze(-1), dim=1) / 1000  # [batch]
        fixed_cost_reg = torch.sum(regular * torch.sum(fixed_cost_point, dim=1), dim=1)  # [batch]

        # calculate food stores
        u_sum_food = torch.bmm(u_lj_food.transpose(1, 2), food.unsqueeze(-1)).squeeze(-1)
        P_sum_food = u_sum_food + u0 + 1e-8  # add numerical stability constant
        # use torch.clamp to ensure denominator is not too small
        P_sum_food = torch.clamp(P_sum_food, min=1e-6)
        revenue_f_block = torch.sum(pop.squeeze(-1) * (u_sum_food / P_sum_food) * self.lambda_f, dim=1)  # [batch]
        delivery_cost_food = self.tau * torch.sum(food * delivery_distances_food_set.squeeze(-1), dim=1) / 1000  # [batch]
        fixed_cost_food = torch.sum(food * torch.sum(fixed_cost_point, dim=1), dim=1)  # [batch]

        # combine
        revenue_combined = revenue_g_block + revenue_f_block
        delivery_combined = delivery_cost_reg + delivery_cost_food
        fixed_cost_combined = fixed_cost_reg + fixed_cost_food

        labor_cost = 0.075 * revenue_combined
        land_cost = 0.005 * revenue_combined

        reward = revenue_combined * self.miu_nu_other - labor_cost - land_cost - delivery_combined - fixed_cost_combined
        return reward

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # TODO: check solution validity
        pass

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        # TODO: local search
        pass

    @staticmethod
    def get_num_starts(td):
        return td["action_mask"].shape[-1]

    @staticmethod
    def select_start_nodes(td, num_starts):
        num_loc = td["action_mask"].shape[-1]
        return (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
        )
