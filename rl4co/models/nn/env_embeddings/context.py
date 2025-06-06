import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.utils.ops import batched_scatter_sum, gather_by_index


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPContext,
        "atsp": TSPContext,
        "cvrp": VRPContext,
        "cvrptw": VRPTWContext,
        "cvrpmvc": VRPContext,
        "ffsp": FFSPContext,
        "svrp": SVRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "spctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
        "mdpp": DPPContext,
        "pdp": PDPContext,
        "mdcpdp": MDCPDPContext,
        "mtsp": MTSPContext,
        "smtwtp": SMTWTPContext,
        "mtvrp": MTVRPContext,
        "shpp": TSPContext,
        "flp": FLPContext,
        "mcp": MCPContext,
        "wal": WALContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class FFSPContext(EnvContext):
    def __init__(self, embed_dim, stage_cnt=None):
        self.has_stage_emb = stage_cnt is not None
        step_context_dim = (1 + int(self.has_stage_emb)) * embed_dim
        super().__init__(embed_dim=embed_dim, step_context_dim=step_context_dim)
        if self.has_stage_emb:
            self.stage_emb = nn.Parameter(torch.rand(stage_cnt, embed_dim))

    def _cur_node_embedding(self, embeddings: TensorDict, td):
        cur_node_embedding = gather_by_index(
            embeddings["machine_embeddings"], td["stage_machine_idx"]
        )
        return cur_node_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        if self.has_stage_emb:
            state_embedding = self._state_embedding(embeddings, td)
            context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
            return self.project_context(context_embedding)
        else:
            return self.project_context(cur_node_embedding)

    def _state_embedding(self, _, td):
        cur_stage_emb = self.stage_emb[td["stage_idx"]]
        return cur_stage_emb


class TSPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(TSPContext, self).__init__(embed_dim, 2 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embed_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 1
        )

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding


class VRPTWContext(VRPContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
        - current time
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )

    def _state_embedding(self, embeddings, td):
        capacity = super()._state_embedding(embeddings, td)
        current_time = td["current_time"]
        return torch.cat([capacity, current_time], -1)


class SVRPContext(EnvContext):
    """Context embedding for the Skill Vehicle Routing Problem (SVRP).
    Project the following to the embedding space:
        - current node embedding
        - current technician
    """

    def __init__(self, embed_dim):
        super(SVRPContext, self).__init__(embed_dim=embed_dim, step_context_dim=embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class PCTSPContext(EnvContext):
    """Context embedding for the Prize Collecting TSP (PCTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining prize (prize_required - cur_total_prize)
    """

    def __init__(self, embed_dim):
        super(PCTSPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = torch.clamp(
            td["prize_required"] - td["cur_total_prize"], min=0
        )[..., None]
        return state_embedding


class OPContext(EnvContext):
    """Context embedding for the Orienteering Problem (OP).
    Project the following to the embedding space:
        - current node embedding
        - remaining distance (max_length - tour_length)
    """

    def __init__(self, embed_dim):
        super(OPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["max_length"][..., 0] - td["tour_length"]
        return state_embedding[..., None]


class DPPContext(EnvContext):
    """Context embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Project the following to the embedding space:
        - current cell embedding
    """

    def __init__(self, embed_dim):
        super(DPPContext, self).__init__(embed_dim)

    def forward(self, embeddings, td):
        """Context cannot be defined by a single node embedding for DPP, hence 0.
        We modify the dynamic embedding instead to capture placed items
        """
        return embeddings.new_zeros(embeddings.size(0), self.embed_dim)


class PDPContext(EnvContext):
    """Context embedding for the Pickup and Delivery Problem (PDP).
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(PDPContext, self).__init__(embed_dim, embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class MTSPContext(EnvContext):
    """Context embedding for the Multiple Traveling Salesman Problem (mTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining_agents
        - current_length
        - max_subtour_length
        - distance_from_depot
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(MTSPContext, self).__init__(embed_dim, 2 * embed_dim)
        proj_in_dim = (
            4  # remaining_agents, current_length, max_subtour_length, distance_from_depot
        )
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding.squeeze()

    def _state_embedding(self, embeddings, td):
        dynamic_feats = torch.stack(
            [
                (td["num_agents"] - td["agent_idx"]).float(),
                td["current_length"],
                td["max_subtour_length"],
                self._distance_from_depot(td),
            ],
            dim=-1,
        )
        return self.proj_dynamic_feats(dynamic_feats)

    def _distance_from_depot(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 0, :], dim=-1)


class SMTWTPContext(EnvContext):
    """Context embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Project the following to the embedding space:
        - current node embedding
        - current time
    """

    def __init__(self, embed_dim):
        super(SMTWTPContext, self).__init__(embed_dim, embed_dim + 1)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_job"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["current_time"]
        return state_embedding


class MDCPDPContext(EnvContext):
    """Context embedding for the MDCPDP.
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(MDCPDPContext, self).__init__(embed_dim, embed_dim * 2 + 5)

    def _state_embedding(self, embeddings, td):
        # get number of visited cities over total
        num_agents = td["capacity"].shape[-1]
        num_cities = td["locs"].shape[-2] - num_agents
        unvisited_number = td["available"][..., num_agents:].sum(-1)
        agent_capacity = td["capacity"].gather(-1, td["current_depot"])
        current_to_deliver = td["to_deliver"][..., num_agents + num_cities // 2 :]

        context_feats = torch.cat(
            [
                agent_capacity - td["current_carry"],  # current available capacity
                td["current_length"].gather(-1, td["current_depot"]),
                unvisited_number[..., None] / num_cities,
                current_to_deliver.sum(-1)[..., None],  # current to deliver number
                td["current_length"].max(-1)[0][..., None],  # max length
            ],
            -1,
        )
        return context_feats

    def _cur_agent_embedding(self, embeddings, td):
        """Get embedding of current agent"""
        cur_agent_embedding = gather_by_index(embeddings, td["current_depot"])
        return cur_agent_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        cur_agent_embedding = self._cur_agent_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat(
            [cur_node_embedding, cur_agent_embedding, state_embedding], -1
        )
        return self.project_context(context_embedding)


class SchedulingContext(nn.Module):
    def __init__(self, embed_dim: int, scaling_factor: int = 1000):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.proj_busy = nn.Linear(1, embed_dim, bias=False)

    def forward(self, h, td):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        busy_proj = self.proj_busy(busy_for.unsqueeze(-1))
        # (b m e)
        return h + busy_proj


class MTVRPContext(VRPContext):
    """Context embedding for Multi-Task VRPEnv.
    Project the following to the embedding space:
        - current node embedding
        - remaining_linehaul_capacity (vehicle_capacity - used_capacity_linehaul)
        - remaining_backhaul_capacity (vehicle_capacity - used_capacity_backhaul)
        - current time
        - current_route_length
        - open route indicator
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 5
        )

    def _state_embedding(self, embeddings, td):
        remaining_linehaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_linehaul"]
        )
        remaining_backhaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        current_time = td["current_time"]
        current_route_length = td["current_route_length"]
        open_route = td["open_route"]
        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                current_route_length,
                open_route,
            ],
            -1,
        )


class FLPContext(EnvContext):
    """Context embedding for the Facility Location Problem (FLP)."""

    def __init__(self, embed_dim: int):
        super(FLPContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, embeddings, td):
        cur_dist = td["distances"].unsqueeze(-2)  # (batch_size, 1, n_points)
        dist_improve = cur_dist - td["orig_distances"]  # (batch_size, n_points, n_points)
        dist_improve = torch.clamp(dist_improve, min=0).sum(-1)  # (batch_size, n_points)

        # softmax
        loc_best_soft = torch.softmax(dist_improve, dim=-1)  # (batch_size, n_points)
        context_embedding = (embeddings * loc_best_soft[..., None]).sum(-2)
        return self.project_context(context_embedding)


class MCPContext(EnvContext):
    """Context embedding for the Maximum Coverage Problem (MCP)."""

    def __init__(self, embed_dim: int):
        super(MCPContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, embeddings, td):
        membership_weighted = batched_scatter_sum(
            td["weights"].unsqueeze(-1), td["membership"].long()
        )
        membership_weighted.squeeze_(-1)
        # membership_weighted: [batch_size, n_sets]

        # softmax; higher weights for better sets
        membership_weighted = torch.softmax(
            membership_weighted, dim=-1
        )  # (batch_size, n_sets)
        context_embedding = (membership_weighted.unsqueeze(-1) * embeddings).sum(1)
        return self.project_context(context_embedding)


class WALContext(EnvContext):
    """Context embedding for the Walmart Location Problem (WAL).
    Project the following to the embedding space:
        - revenue improvement potential for each location
        - based on precomputed WAL features and current store selection state
    """

    def __init__(self, embed_dim: int, chunk_size: int = 32):
        super(WALContext, self).__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size  # Configurable chunk size for memory efficiency
        self.project_context = nn.Linear(embed_dim, embed_dim, bias=True)
        # WAL-specific parameters (matching the environment)
        self.lambda_g, self.lambda_f, self.tau, self.miu_nu_other = 1.938, 1.912, 3.5, 0.19

    def forward(self, embeddings, td):
        """
        Compute context embedding for WAL problem based on COMPLETE reward improvement potential.
        Includes revenue, delivery costs, fixed costs, labor costs, and land costs.
        Similar to FLPContext but focused on full reward improvement rather than just revenue improvement.
        Memory-efficient implementation using chunked processing.
        """
        chosen_count = td["chosen_count"]  # [batch, n_points]
        batch_size, n_points = chosen_count.shape
        
        # Check if WAL features are present in td, if not they should have been computed in reset
        if "u_lj_reg" not in td:
            raise RuntimeError("WAL features not found in TensorDict. Make sure the environment computes them during reset.")
        
        # Get WAL features (could be sparse or dense depending on device compatibility)
        u_lj_reg_input = td["u_lj_reg"]    # [batch, N, block_num] - could be sparse or dense
        u_lj_food_input = td["u_lj_food"]  # [batch, N, block_num] - could be sparse or dense
        u0 = td["u0"]                # [batch, block_num]
        pop = td["pop"]              # [batch, block_num, 1]
        delivery_distances_reg_set = td["delivery_distances_reg_set"]  # [batch, N, 1]
        delivery_distances_food_set = td["delivery_distances_food_set"]  # [batch, N, 1]
        fixed_cost_point = td["fixed_cost_point"]  # [batch, 1, N]
        
        # Convert to dense if sparse for computation
        if hasattr(u_lj_reg_input, 'is_sparse') and u_lj_reg_input.is_sparse:
            u_lj_reg = u_lj_reg_input.to_dense()
        else:
            u_lj_reg = u_lj_reg_input
            
        if hasattr(u_lj_food_input, 'is_sparse') and u_lj_food_input.is_sparse:
            u_lj_food = u_lj_food_input.to_dense()
        else:
            u_lj_food = u_lj_food_input
        
        # Calculate current COMPLETE reward (baseline) - including ALL costs
        regular = (chosen_count >= 1).float()  # [batch, n_points]
        food = (chosen_count >= 2).float()     # [batch, n_points]

        regular_camdidate = 1 - regular
        food_camdidate = regular - food

        diag_regular = torch.diag_embed(regular_camdidate)
        diag_food = torch.diag_embed(food_camdidate)

        u_reg = torch.bmm(u_lj_reg.transpose(1, 2), diag_regular)
        P_reg = u0.unsqueeze(-1) + u_reg + 1e-8   # Ensure u0 is dense
        revenue_g_candidate = torch.sum(pop * (u_reg / P_reg) * self.lambda_g, dim=1).squeeze(1).view(batch_size, n_points)
        
        u_food = torch.bmm(u_lj_food.transpose(1, 2), diag_food)
        P_food = u0.unsqueeze(-1) + u_food + 1e-8
        revenue_f_candidate = torch.sum(pop * (u_food / P_food) * self.lambda_f, dim=1).squeeze(1).view(batch_size, n_points)

        # Calculate costs
        delivery_cost_reg_candidate = self.tau * (regular_camdidate * delivery_distances_reg_set.squeeze(-1)) / 1000
        fixed_cost_reg_candidate = regular_camdidate * fixed_cost_point.squeeze(1).view(batch_size, n_points)
        delivery_cost_food_candidate = self.tau * (food_camdidate * delivery_distances_food_set.squeeze(-1)) / 1000
        fixed_cost_food_candidate = food_camdidate * fixed_cost_point.squeeze(1).view(batch_size, n_points)
        
        revenue_combined_candidate = revenue_g_candidate + revenue_f_candidate
        delivery_combined_candidate = delivery_cost_reg_candidate + delivery_cost_food_candidate
        fixed_cost_combined_candidate = fixed_cost_reg_candidate + fixed_cost_food_candidate
        
        # Calculate labor and land costs
        labor_cost_reg = 0.075 * revenue_combined_candidate
        land_cost_reg = 0.005 * revenue_combined_candidate

        advantage_candidate = revenue_combined_candidate * self.miu_nu_other - labor_cost_reg - land_cost_reg - delivery_combined_candidate - fixed_cost_combined_candidate
        
        reward_improve = torch.clamp(advantage_candidate, min=0) / 100.0
        loc_best_soft = torch.softmax(reward_improve, dim=-1)  # [batch, n_points]

        # Weighted combination of node embeddings (same as FLPContext)
        context_embedding = (embeddings * loc_best_soft[..., None]).sum(-2)
        
        return self.project_context(context_embedding)