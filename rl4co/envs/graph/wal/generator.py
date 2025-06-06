import math

from typing import Callable

import torch

import pandas as pd

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler

from rl4co.utils.pylogger import get_pylogger


log = get_pylogger(__name__)

def haversine_torch(lat1, lon1, lat2, lon2):
    """
    lat1, lon1: [batch, N]
    lat2, lon2: [M]
    return: [batch, N, M]
    """
    R = 3958.8
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)
    # [batch, N, 1] - [1, 1, M] => [batch, N, M]
    dlat = lat2[None, None, :] - lat1[:, :, None]
    dlon = lon2[None, None, :] - lon1[:, :, None]
    a = torch.sin(dlat/2)**2 + torch.cos(lat1[:, :, None]) * torch.cos(lat2[None, None, :]) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    return R * c

def haversine_torch_ultra_efficient(lat1, lon1, lat2, lon2, max_distance=30, chunk_size=500):
    """
    Ultra memory-efficient version that returns only nearby blocks
    Returns: (distances, valid_indices) where distances are only for nearby blocks
    """
    R = 3958.8
    batch_size, N = lat1.shape
    M = lat2.shape[0]
    device = lat1.device
    
    # Convert to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    # Store results as lists and then concat
    all_distances = []
    all_batch_idx = []
    all_point_idx = []
    all_block_idx = []
    
    # Process in chunks to save memory
    for start_idx in range(0, M, chunk_size):
        end_idx = min(start_idx + chunk_size, M)
        
        # Current chunk
        lat2_chunk = lat2_rad[start_idx:end_idx]
        lon2_chunk = lon2_rad[start_idx:end_idx]
        
        # Calculate distances for this chunk
        dlat = lat2_chunk[None, None, :] - lat1_rad[:, :, None]
        dlon = lon2_chunk[None, None, :] - lon1_rad[:, :, None]
        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad[:, :, None]) * torch.cos(lat2_chunk[None, None, :]) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        chunk_distances = R * c
        
        # Find valid distances (within threshold)
        mask = chunk_distances <= max_distance
        
        if mask.any():
            # Get indices of valid distances
            batch_indices, point_indices, block_indices_rel = torch.where(mask)
            block_indices_abs = block_indices_rel + start_idx
            
            # Store the valid distances and their indices
            valid_distances = chunk_distances[mask]
            
            all_distances.append(valid_distances)
            all_batch_idx.append(batch_indices)
            all_point_idx.append(point_indices) 
            all_block_idx.append(block_indices_abs)
        
        # Clean up
        del dlat, dlon, a, c, chunk_distances, mask
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    if all_distances:
        # Concatenate all results
        distances = torch.cat(all_distances)
        batch_idx = torch.cat(all_batch_idx)
        point_idx = torch.cat(all_point_idx)
        block_idx = torch.cat(all_block_idx)
        
        # Create sparse representation - try to create directly on target device
        indices = torch.stack([batch_idx, point_idx, block_idx])
        
        # Try to create sparse tensor directly on target device
        try:
            sparse_distances = torch.sparse_coo_tensor(
                indices, distances, (batch_size, N, M), device=device
            ).coalesce()
            # print(f"✅ Sparse tensor created directly on {device}")
        except Exception as e:
            print(f"⚠️  Warning: Direct sparse tensor creation failed on {device}, using CPU fallback. Error: {e}")
            # Create sparse tensor on CPU first
            indices_cpu = indices.cpu()
            distances_cpu = distances.cpu()
            sparse_distances = torch.sparse_coo_tensor(
                indices_cpu, distances_cpu, (batch_size, N, M), device='cpu'
            ).coalesce()
            
            # Move to target device
            if device.type != 'cpu':
                try:
                    sparse_distances = sparse_distances.to(device)
                    print(f"✅ Sparse tensor moved to {device}")
                except Exception as e2:
                    print(f"❌ Warning: Could not move sparse tensor to {device}. Error: {e2}")
                    print("Falling back to dense tensor on target device...")
                    # Convert to dense and move to device as fallback
                    dense_tensor = sparse_distances.to_dense().to(device)
                    return dense_tensor
        
        return sparse_distances
    else:
        # No valid distances found, return empty sparse tensor directly on target device
        try:
            indices = torch.zeros((3, 0), dtype=torch.long, device=device)
            values = torch.zeros(0, device=device)
            sparse_tensor = torch.sparse_coo_tensor(indices, values, (batch_size, N, M), device=device)
            print(f"✅ Empty sparse tensor created directly on {device}")
        except Exception as e:
            print(f"⚠️  Warning: Direct empty sparse tensor creation failed on {device}, using CPU fallback. Error: {e}")
            indices = torch.zeros((3, 0), dtype=torch.long, device='cpu')
            values = torch.zeros(0, device='cpu')
            sparse_tensor = torch.sparse_coo_tensor(indices, values, (batch_size, N, M), device='cpu')
            
            # Try to move to target device
            if device.type != 'cpu':
                try:
                    sparse_tensor = sparse_tensor.to(device)
                    print(f"✅ Empty sparse tensor moved to {device}")
                except Exception as e2:
                    print(f"❌ Warning: Could not move empty sparse tensor to {device}. Error: {e2}")
                    # Return zero dense tensor as fallback
                    return torch.zeros((batch_size, N, M), device=device)
        
        return sparse_tensor

class GPUFriendlySampler:
    """Base class for samplers that support GPU-native tensor generation"""
    
    def sample(self, shape, device=None):
        """Sample tensors directly on the specified device"""
        raise NotImplementedError("Subclasses must implement sample method")

class NYBlockSampler(GPUFriendlySampler):
    def __init__(self, coords_norm):
        self.coords_norm = coords_norm

    def sample(self, shape, device=None):
        # shape: (*batch_size, num_loc, 2)
        batch_shape = shape[:-2]
        num_loc = shape[-2]
        N = self.coords_norm.shape[0]
        
        # Determine target device
        if device is None:
            device = self.coords_norm.device
        
        # sample batch_size * num_loc indices on target device
        total = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
        idx = torch.randint(0, N, (total, num_loc), device=device)
        
        # Move coords_norm to target device if needed
        coords_norm_device = self.coords_norm.to(device)
        
        # sample
        samples = coords_norm_device[idx]  # shape: (total, num_loc, 2)
        # reshape back to batch_shape
        if batch_shape:
            samples = samples.view(*batch_shape, num_loc, 2)
        return samples


class WALGenerator(Generator):
    """Data generator for the Facility Location Problem (FLP) with Walmart-specific optimizations.
    
    Memory-Efficient Dynamic Computation:
    - Dataset generation only creates basic location data (locs, chosen, to_choose)
    - WAL features (u_lj_reg, u_lj_food, etc.) are computed dynamically during environment reset
    - This approach drastically reduces memory usage during dataset creation
    
    Memory Efficiency Comparison:
    Original approach (all features precomputed):
    - 10,000 batches × WAL features ≈ 50-100GB RAM
    
    New dynamic approach:
    - 10,000 batches × basic data ≈ 1-5GB RAM  
    - WAL features computed per batch during training ≈ 100-500MB per batch
    
    Memory Efficiency Modes (for WAL feature computation):
    - "none": Original implementation, returns dense tensors (~100GB for large datasets)
    - "chunked": Process blocks in chunks with distance filtering, returns sparse tensors (recommended, ~10-20GB)  
    - "ultra": Sparse tensor implementation throughout, returns sparse tensors (lowest memory usage, ~5-10GB)
    
    Note: WAL features are computed on-demand in the environment's reset() method.
    When memory_efficient != "none", u_lj_reg and u_lj_food are returned as sparse tensors
    to save memory. They are automatically converted to dense tensors in _get_reward() and WALContext
    when needed for computation.
    
    Example usage:
        # Memory-efficient training (recommended)
        env = WALEnv(generator_params=dict(memory_efficient="ultra"))
        
        # Even more memory-efficient for very large datasets
        env = WALEnv(generator_params=dict(memory_efficient="ultra"))
        
        # Original approach (not recommended for large datasets)
        env = WALEnv(generator_params=dict(memory_efficient="none"))

        # GPU-native generation (avoids CPU-GPU transfers)
        env = WALEnv(generator_params=dict(device="cuda"))

    Args:
        num_loc: number of locations in the FLP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        memory_efficient: Memory optimization level ("none", "chunked", "ultra")
        device: Device to generate tensors on ("cpu", "cuda", etc.). If None, uses environment device

    Returns:
        A TensorDict with basic data only (WAL features computed dynamically):
            locs [batch_size, num_loc, 2]: locations
            chosen [batch_size, num_loc]: indicators of chosen locations  
            to_choose [batch_size, 1]: number of locations to choose in the FLP
            
        WAL features added dynamically during environment reset:
            u_lj_reg [batch_size, num_loc, block_num]: utility matrix for regular stores (sparse when memory_efficient != "none")
            u_lj_food [batch_size, num_loc, block_num]: utility matrix for food stores (sparse when memory_efficient != "none")
            u0 [batch_size, block_num]: baseline utility
            delivery_distances_reg_set [batch_size, num_loc, 1]: delivery distances for regular stores
            delivery_distances_food_set [batch_size, num_loc, 1]: delivery distances for food stores
            fixed_cost_point [batch_size, 1, num_loc]: fixed costs per location
            pop [batch_size, block_num, 1]: population data
    """

    def __init__(
        self,
        num_loc: int = 250,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,#TODO
        to_choose: int = 90,
        NY_blocks_without_longisland: int = pd.read_csv('./wal_envdata/NY_block_without_longisland.csv'),
        Walmart_locations_dataset: int = './wal_envdata/walmart730.csv',
        block_NY_path='./wal_envdata/block_NY_information_select.xlsx', 
        regdc_path='./wal_envdata/RegDC_NY.xlsx', 
        fooddc_path='./wal_envdata/FoodDC_NY.xlsx',
        memory_efficient: str = "ultra",  # "none", "chunked", "ultra"
        device: str = None,  # Target device for tensor generation
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.to_choose = to_choose
        self.memory_efficient = memory_efficient
        self.device = device  # Store target device for generation
        
        coords = NY_blocks_without_longisland[['lon', 'lat']].values  # shape: (N, 2)
        # normalize to [0, 1]
        self.min_xy = coords.min(axis=0)
        self.max_xy = coords.max(axis=0)
        coords_norm = (coords - self.min_xy) / (self.max_xy - self.min_xy)
        self.coords_norm = torch.tensor(coords_norm, dtype=torch.float)
        
        # Load block data for WAL calculations
        self.block_NY = pd.read_excel(block_NY_path, sheet_name='Sheet1').to_numpy()
        self.latitude = torch.tensor(self.block_NY[:, 5], dtype=torch.float32)
        self.longitude = torch.tensor(self.block_NY[:, 6], dtype=torch.float32)
        self.RegDC_NY = pd.read_excel(regdc_path, sheet_name='Sheet1').to_numpy()
        self.FoodDC_NY = pd.read_excel(fooddc_path, sheet_name='Sheet1').to_numpy()
        self.RegDC_lat = torch.tensor(self.RegDC_NY[:, 0], dtype=torch.float32)
        self.RegDC_lon = torch.tensor(self.RegDC_NY[:, 1], dtype=torch.float32)
        self.FoodDC_lat = torch.tensor(self.FoodDC_NY[:, 0], dtype=torch.float32)
        self.FoodDC_lon = torch.tensor(self.FoodDC_NY[:, 1], dtype=torch.float32)
        
        # 静态人口相关变量
        self.Popden = torch.clamp(torch.tensor(self.block_NY[:, 2] / 1000, dtype=torch.float32), min=1).unsqueeze(1)  # [block_num, 1]
        self.pop = torch.tensor(self.block_NY[:, 1] / 1000, dtype=torch.float32).unsqueeze(1)  # [block_num, 1]
        
        # u0 计算
        alpha0, alpha1, alpha2 = -7.834, 1.861, -0.059
        alpha_pci, alpha_black, alpha_young, alpha_old = 0.013, 0.297, 1.132, 0.465
        pci = torch.clamp(torch.tensor(self.block_NY[:, 7] / 1000, dtype=torch.float32), min=5).unsqueeze(1)
        popblackshare = torch.tensor(self.block_NY[:, 8], dtype=torch.float32).unsqueeze(1)
        popyoungshare = torch.tensor(self.block_NY[:, 9], dtype=torch.float32).unsqueeze(1)
        popoldshare = torch.tensor(self.block_NY[:, 10], dtype=torch.float32).unsqueeze(1)
        
        b_Popden = alpha0 + alpha1 * torch.log(self.Popden) + alpha2 * torch.square(torch.log(self.Popden))
        self.u0 = torch.exp(
            b_Popden
            + alpha_pci * pci
            + alpha_black * popblackshare
            + alpha_young * popyoungshare
            + alpha_old * popoldshare
        ).squeeze(1)  # [block_num]
        
        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = NYBlockSampler(self.coords_norm)
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )
        self.Walmart_locations_dataset = Walmart_locations_dataset

    def _get_generation_device(self, env_device=None):
        """Determine the device for tensor generation"""
        if self.device is not None:
            return torch.device(self.device)
        elif env_device is not None:
            return env_device
        else:
            return torch.device('cpu')

    def test_sample(self, batch_size, Walmart_locations_dataset, min_xy , max_xy, device=None):
        Walmart_locations_dataset = pd.read_csv(Walmart_locations_dataset)
        Walmart_locations_dataset = Walmart_locations_dataset[['lon', 'lat']].values
        Walmart_locations_dataset = (Walmart_locations_dataset - min_xy) / (max_xy - min_xy)
        Walmart_locations_dataset = torch.tensor(Walmart_locations_dataset, dtype=torch.float, device=device)
        locs = Walmart_locations_dataset.unsqueeze(0).expand(batch_size, -1, -1)
        return locs
        
    def _generate(self, batch_size, phase="train", device=None) -> TensorDict:
        # Determine target device
        target_device = self._get_generation_device(device)
        
        # Sample locations directly on target device
        if phase == "test":
            locs = self.test_sample(*batch_size, self.Walmart_locations_dataset, self.min_xy, self.max_xy, device=target_device)
        else:
            # Pass device to sampler for GPU-native sampling
            if hasattr(self.loc_sampler, 'sample') and 'device' in self.loc_sampler.sample.__code__.co_varnames:
                locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2), device=target_device)
            else:
                locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
                if target_device.type != 'cpu':
                    locs = locs.to(target_device)
                
        return TensorDict(
            {
                "locs": locs,
                "chosen": torch.zeros(*batch_size, self.num_loc, dtype=torch.bool, device=target_device),
                "to_choose": torch.ones(*batch_size, dtype=torch.long, device=target_device) * self.to_choose,
                # WAL features will be computed on-demand in the environment
            },
            batch_size=batch_size,
            device=target_device,
        )

    def _precompute_wal_features_ultra_efficient(self, locs, batch_size):
        """Ultra memory-efficient version using sparse tensors"""
        device = locs.device
        batch_size_val = batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size
        num_locs_actual = locs.shape[1]  # Get actual number of locations from input
        
        # Adjust chunk size for MPS compatibility
        chunk_size = 50 if device.type == 'mps' else 500
        
        # Transform normalized coordinates to real coordinates
        max_xy = torch.tensor(self.max_xy, dtype=torch.float32, device=device)
        min_xy = torch.tensor(self.min_xy, dtype=torch.float32, device=device)
        points_data = locs * (max_xy - min_xy) + min_xy
        
        # Get geographic data on correct device
        latitude = self.latitude.to(device)
        longitude = self.longitude.to(device)
        RegDC_lat = self.RegDC_lat.to(device)
        RegDC_lon = self.RegDC_lon.to(device)
        FoodDC_lat = self.FoodDC_lat.to(device)
        FoodDC_lon = self.FoodDC_lon.to(device)
        Popden = self.Popden.to(device)
        pop = self.pop.to(device)
        u0 = self.u0.to(device)
        
        latitude_point = points_data[:, :, 1]  # [batch, N]
        longitude_point = points_data[:, :, 0]  # [batch, N]

        # Use sparse distance calculation
        sparse_distances = haversine_torch_ultra_efficient(latitude_point, longitude_point, latitude, longitude, max_distance=30, chunk_size=chunk_size)
        
        # For delivery distances, still use the original function since these are much smaller
        delivery_distances_reg_set = haversine_torch(latitude_point, longitude_point, RegDC_lat, RegDC_lon)
        delivery_distances_food_set = haversine_torch(latitude_point, longitude_point, FoodDC_lat, FoodDC_lon)
        
        # u_lj_reg, u_lj_food computation
        ksi0, ksi1, gamma = 0.703, -0.056, 0.207
        h_Popden = ksi0 + ksi1 * torch.log(Popden)  # [block_num, 1]
        h_Popden = h_Popden.squeeze(1)  # [block_num]
        
        # Work directly with sparse tensor for efficiency
        if hasattr(sparse_distances, 'is_sparse') and sparse_distances.is_sparse:
            indices = sparse_distances.indices()  # [3, nnz]
            values = sparse_distances.values()    # [nnz]
            
            # Create masks for distance thresholds
            mask_25 = values < 25
            mask_30 = values < 30
            
            # Calculate u_lj values only for valid (sparse) entries
            block_indices = indices[2]  # block indices for each valid distance
            h_values = h_Popden[block_indices]  # corresponding h_Popden values
            
            # FIXED: Use correct formula matching the chunked version
            u_lj_reg_values = torch.exp((-h_values) * values + gamma) * mask_25
            u_lj_food_values = torch.exp((-h_values) * values + gamma) * mask_25
            
            # Create sparse tensors for u_lj - try to create directly on target device, fallback to CPU if needed
            try:
                # Try to create sparse tensors directly on target device first
                u_lj_reg_sparse = torch.sparse_coo_tensor(indices, u_lj_reg_values, sparse_distances.shape, device=device)
                u_lj_food_sparse = torch.sparse_coo_tensor(indices, u_lj_food_values, sparse_distances.shape, device=device)
                # print(f"✅ u_lj sparse tensors created directly on {device}")
            except Exception as e:
                print(f"⚠️  Warning: Direct sparse tensor creation failed on {device}, using CPU fallback. Error: {e}")
                # Always create sparse tensors on CPU first for MPS compatibility
                indices_cpu = indices.cpu()
                u_lj_reg_values_cpu = u_lj_reg_values.cpu()
                u_lj_food_values_cpu = u_lj_food_values.cpu()
                
                # Create sparse tensors on CPU
                u_lj_reg_sparse_cpu = torch.sparse_coo_tensor(indices_cpu, u_lj_reg_values_cpu, sparse_distances.shape, device='cpu')
                u_lj_food_sparse_cpu = torch.sparse_coo_tensor(indices_cpu, u_lj_food_values_cpu, sparse_distances.shape, device='cpu')
                
                # Try to move to target device
                try:
                    u_lj_reg_sparse = u_lj_reg_sparse_cpu.to(device)
                    u_lj_food_sparse = u_lj_food_sparse_cpu.to(device)
                    print(f"✅ u_lj sparse tensors moved to {device}")
                except Exception as e2:
                    print(f"❌ Warning: Sparse tensor creation failed on {device}, using dense tensors. Error: {e2}")
                    # Convert sparse_distances to dense and use regular computation
                    distances_dense = sparse_distances.to_dense() if hasattr(sparse_distances, 'to_dense') else sparse_distances
                    mask_25_dense = distances_dense < 25
                    u_lj_reg_sparse = torch.exp((-h_Popden)[None, None, :] * distances_dense + gamma) * mask_25_dense
                    u_lj_food_sparse = torch.exp((-h_Popden)[None, None, :] * distances_dense + gamma) * mask_25_dense
            
            # FIXED: Use efficient scatter operation instead of loop for fixed_cost_point
            # Create a mask for 30-mile distances
            indices_30 = indices[:, mask_30]  # [3, nnz_30]
            
            # Use scatter_add for efficient accumulation
            pop_store = torch.zeros(batch_size_val, num_locs_actual, 1, device=device)  # FIXED: Use actual num_locs
            if indices_30.shape[1] > 0:  # Only if there are valid distances
                batch_idx = indices_30[0]  # [nnz_30]
                point_idx = indices_30[1]  # [nnz_30]
                block_idx = indices_30[2]  # [nnz_30]
                
                # Get corresponding population values
                pop_values_30 = pop[block_idx].squeeze()  # [nnz_30]
                
                # FIXED: Create indices for scatter_add using actual num_locs
                linear_indices = batch_idx * num_locs_actual + point_idx
                pop_store_flat = pop_store.view(-1, 1)
                
                # Use scatter_add to accumulate population
                pop_store_flat.scatter_add_(0, linear_indices.unsqueeze(1), pop_values_30.unsqueeze(1))
                pop_store = pop_store_flat.view(batch_size_val, num_locs_actual, 1)
        else:
            # Fallback to dense computation when sparse tensor creation failed
            print(f"Warning: Working with dense tensors due to sparse tensor limitations on {device}")
            distances_dense = sparse_distances
            mask_25 = distances_dense < 25
            mask_30 = distances_dense < 30
            
            u_lj_reg_sparse = torch.exp((-h_Popden)[None, None, :] * distances_dense + gamma) * mask_25
            u_lj_food_sparse = torch.exp((-h_Popden)[None, None, :] * distances_dense + gamma) * mask_25
            
            # Calculate pop_store using dense operations
            pop_store = torch.sum(pop[None, :, :] * mask_30.transpose(1, 2), dim=1, keepdim=True)
        
        pop_store = torch.clamp(pop_store, min=1e-6)
        omega1, omega2 = 1.5244015610139994, -0.13384045894730612
        fixed_cost_point = omega1 * torch.log(pop_store) + omega2 * torch.square(torch.log(pop_store))
        fixed_cost_point = fixed_cost_point.transpose(1, 2)  # [batch, 1, N]
        
        # Expand static data to match batch size
        u0_expanded = u0.unsqueeze(0).expand(batch_size_val, -1)  # [batch, block_num]
        pop_expanded = pop.unsqueeze(0).expand(batch_size_val, -1, -1)  # [batch, block_num, 1]
        
        return {
            "u_lj_reg": u_lj_reg_sparse,  # [batch, N, block_num] - sparse or dense depending on compatibility
            "u_lj_food": u_lj_food_sparse,  # [batch, N, block_num] - sparse or dense depending on compatibility
            "u0": u0_expanded,  # [batch, block_num]
            "delivery_distances_reg_set": delivery_distances_reg_set.min(dim=-1, keepdim=True).values,  # [batch, N, 1]
            "delivery_distances_food_set": delivery_distances_food_set.min(dim=-1, keepdim=True).values,  # [batch, N, 1] 
            "fixed_cost_point": fixed_cost_point,  # [batch, 1, N]
            "pop": pop_expanded,  # [batch, block_num, 1]
        }

    def compute_wal_features_on_demand(self, locs, batch_size):
        """
        Compute WAL features on demand for a given batch of locations.
        This is called from the environment during reset to compute features dynamically.
        """
        device = locs.device
        
        # Ensure all static data is on the same device as locs to avoid transfers
        self._move_static_data_to_device(device)
        
        # On MPS, use simple "none" mode to avoid dimension and sparse tensor issues
        if device.type == 'mps':
            print("⚠️  Info: MPS detected - using 'none' mode to avoid compatibility issues")
            return self._precompute_wal_features(locs, batch_size)
        elif self.memory_efficient == "ultra":
            return self._precompute_wal_features_ultra_efficient(locs, batch_size)
        else:
            return self._precompute_wal_features(locs, batch_size)
    
    def _move_static_data_to_device(self, device):
        """Move all static data tensors to the target device if not already there"""
        if self.latitude.device != device:
            print(f"📡 Moving static data from {self.latitude.device} to {device}")
            self.latitude = self.latitude.to(device)
            self.longitude = self.longitude.to(device)
            self.RegDC_lat = self.RegDC_lat.to(device)
            self.RegDC_lon = self.RegDC_lon.to(device)
            self.FoodDC_lat = self.FoodDC_lat.to(device)
            self.FoodDC_lon = self.FoodDC_lon.to(device)
            self.Popden = self.Popden.to(device)
            self.pop = self.pop.to(device)
            self.u0 = self.u0.to(device)
            self.coords_norm = self.coords_norm.to(device)
            print(f"✅ Static data moved to {device}")

