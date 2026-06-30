# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, dtype_override=None, temperature=0.001, freq_coeff=None, freq_scaling = 1, pe_type='transformer',  fone_dim = 8 * 3, pe_scaling = 1):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.

        Args:
            channels (int): The number of channels for the positional encoding.
            dtype_override (torch.dtype, optional): If set, overrides the dtype of the output embedding
            temperature (float, optional): Temperature for softmax normalization in decode_pos. Default is 0.001.
            freq_coeff (float, optional): Frequency coefficient for the positional encoding.
            pe_type (str, optional): Type of positional encoding. 'transformer' or 'nerf'. Default is 'transformer'.
            fone_dim
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1

        self.pe_type = pe_type  # 'transformer' or 'nerf'
        if self.pe_type not in ['transformer', 'nerf', 'fone']:
            raise ValueError("pe_type must be either 'transformer' or 'nerf' or 'fone'!")
        
        if fone_dim % 2 != 0 or fone_dim % 3 != 0:
            raise ValueError("fone_dim must be divied by 3 for 'x,y,z' and must be devided by 2 for 'consine, sine'")
        else:
            self.fone_dim = fone_dim // 3 # 3 for x, y, z
        
        if freq_coeff is None:
            if self.pe_type == 'transformer':
                freq_coeff = 10000.0
            elif self.pe_type == 'nerf':
                freq_coeff = 10 # this is used in nerf
        
        if self.pe_type == 'transformer':
            if freq_coeff <= 50:
                raise ValueError("freq_coeff might be wrong for transformer positional encoding!")
            inv_freq = 1.0 * freq_scaling / (freq_coeff ** (torch.arange(0, channels, 2).float() / channels)) # from 1 to 1/freq_coeff
            self.register_buffer("freq", inv_freq)
        elif self.pe_type == 'nerf': # not compatible with cosine similarity decoding
            if freq_coeff > 50:
                raise ValueError("freq_coeff might be wrong for nerf positional encoding!")
            nerf_freq = 2 ** (freq_coeff * torch.arange(0, channels, 2).float() / channels) * torch.pi # from 1 to 2^freq_coeff * np.pi
            self.register_buffer("freq", nerf_freq)
        elif self.pe_type == 'fone':
            # fone_freq = 2 * torch.pi / (freq_coeff ** (1/2 * torch.arange(-self.fone_dim/2, self.fone_dim/2, 2).float())) # from 2pi to 2pi/freq_coeff**channels (eventually 0) check https://arxiv.org/html/2502.09741v1
            # fone_freq = 2 * torch.pi / (freq_coeff ** (1/2 * torch.arange(0, self.fone_dim, 2).float()))
            fone_freq = 2 * torch.pi * freq_scaling / (freq_coeff ** (1/2 * torch.arange(0, self.fone_dim , 2).float()))
            self.register_buffer("freq", fone_freq) #


        print('freq', self.freq, 'This is the frequency for positional encoding')


        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

        self.temperature = temperature # for softmax normalization in decode_pos (recommanded value is l2: 0.1, cosine: 0.001)
        self.pe_scaling = pe_scaling


    
    def forward(self, tensor):
        """
        :param tensor: A tensor of size (batch_size, num_pixels, 3)
        :return: Positional Encoding Matrix of size (batch_size, num_pixels, channels)
        """
        if len(tensor.shape) != 3 or tensor.shape[-1] != 3:
            raise RuntimeError("The input tensor has to be 3d with last dimension of size 3!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, num_pixels, _ = tensor.shape


        pos_x = tensor[:, :, 0] # shape (batch_size, num_pixels)
        pos_y = tensor[:, :, 1]
        pos_z = tensor[:, :, 2]


        sin_inp_x = torch.einsum("b n, d -> b n d", pos_x, self.freq) # shape (batch_size, num_pixels, channels/2)
        sin_inp_y = torch.einsum("b n, d -> b n d", pos_y, self.freq)
        sin_inp_z = torch.einsum("b n, d -> b n d", pos_z, self.freq)


        emb_x = get_emb(sin_inp_x) # shape (batch_size, num_pixels, channels)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (batch_size, num_pixels, self.channels * 3),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )

        if self.pe_type == 'fone':
            emb[:,:, :self.fone_dim] = emb_x
            emb[:,:, self.fone_dim:2*self.fone_dim] = emb_y
            emb[:,:, 2*self.fone_dim:3*self.fone_dim] = emb_z
        else:
            emb[:,:, :self.channels] = emb_x
            emb[:,:, self.channels:2*self.channels] = emb_y
            emb[:,:, 2*self.channels:] = emb_z

        self.cached_penc = emb
        
        # print(f"Positional Encoding shape: {self.cached_penc.shape}")

        # NOTE: Return with original channels
        self.cached_penc = self.cached_penc[:, :, :self.org_channels]

        # NOTE: scale the positional encoding if needed
        self.cached_penc = self.cached_penc * self.pe_scaling

        return self.cached_penc
    
    def pos_grid_3d(self, pc_range, voxel_size):
        """
        Generate a 3D positional grid.
        :param pc_range: The range of the point cloud. e.g. [x_min, y_min, z_min, x_max, y_max, z_max].
        :param voxel_size: The size of the a voxel in grid. e.g. [x_grid, y_grid, z_grid].
        :return: A 3D grid of positional encodings. shape (x_steps, y_steps, z_steps, channels)
        """

        if len(pc_range) != 6 or len(voxel_size) != 3:
            raise RuntimeError("pc_range must be of size 6 and voxel_size must be of size 3!")

        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        x_grid, y_grid, z_grid = voxel_size
        x = torch.arange(x_min, x_max, x_grid, device=self.freq.device, dtype=self.freq.dtype)
        y = torch.arange(y_min, y_max, y_grid, device=self.freq.device, dtype=self.freq.dtype)
        z = torch.zeros((1), device=self.freq.device, dtype=self.freq.dtype)

        grid_coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1) # shape (x_grids, y_grids, z_grids, 3)

        grid_pos_flatted = grid_coords.reshape(-1, 3)  # shape (x_grids * y_grids * z_grids, 3)
        if self.pe_type == 'fone':
            pos_emb_grid = self.forward(grid_pos_flatted.unsqueeze(0))[:, :, :self.fone_dim * 3]  # shape (1, x_grids * y_grids * z_grids, fone_dim * 3)
            pos_emb_grid = pos_emb_grid.reshape(grid_coords.shape[0], grid_coords.shape[1], grid_coords.shape[2], self.fone_dim * 3)
        else:
            pos_emb_grid = self.forward(grid_pos_flatted.unsqueeze(0))  # shape (1, x_grids * y_grids * z_grids, channels * 3)
            pos_emb_grid = pos_emb_grid.reshape(grid_coords.shape[0], grid_coords.shape[1], grid_coords.shape[2], self.org_channels)

        # NOTE: Return with original channels
        pos_emb_grid = pos_emb_grid[:, :, :, :self.org_channels]

        return pos_emb_grid.detach()  # shape (x_grids, y_grids, z_grids, channels * 3)

    def decode_pos(self, pos_emb, pos_emb_grid, pc_range, voxel_size, sim_method='cosine',  force_temperature=None):
        """
        Decode the positional encoding to get the original position.
        :param 
            pos_emb: A tensor of size (B, num_coords, channels)
            pos_emb_grid: A tensor of size (x_steps, y_steps, z_steps, channels) with the positional encoding grid.
            pc_range: The range of the point cloud. e.g. [x_min, y_min, z_min, x_max, y_max, z_max].
            voxel_size: The size of a voxel in grid. e.g. [x_grid, y_grid, z_grid].
        :return: A tensor of size (num_coords, 3) with the original position.
        """

        if sim_method not in ['cosine', 'l2']:
            raise RuntimeError("sim_method must be either 'cosine' or 'l2'!")
        if sim_method == 'l2':
            self.temperature = 0.1 # for l2 loss, use 0.1 as temperature
        elif sim_method == 'cosine':
            self.temperature = 0.001
        if force_temperature is not None:
            self.temperature = force_temperature


        pos_emb_grid = pos_emb_grid.to(pos_emb.device)
        # print(f"pos_emb_grid shape: {pos_emb_grid.shape}")
        B = pos_emb.shape[0]
        NUM_COORDS = pos_emb.shape[1]
        if len(pos_emb.shape) != 3 or pos_emb.shape[-1] != self.org_channels:
            raise RuntimeError("The input tensor has to be 3d with last dimension of size org_channels!")

        # NOTE: 1. use cosine similarity to find the closest position in the grid
        pos_emb_flat = pos_emb.view(-1, self.org_channels)
        pos_emb_grid_flat = pos_emb_grid.view(-1, self.org_channels)

        # THIS takes too many vram, do it iteratively for every coordinate
        # sim = torch.nn.functional.cosine_similarity(pos_emb_flat.unsqueeze(1), pos_emb_grid_flat.unsqueeze(0), dim=-1) # shape (B, num_coords, x_steps * y_steps * z_steps)
        sim = torch.zeros((B, NUM_COORDS, pos_emb_grid_flat.shape[0]), device=pos_emb.device, dtype=pos_emb.dtype)  # shape (B, num_coords, x_steps * y_steps * z_steps)
        if sim_method == 'cosine':
            for i in range(NUM_COORDS):
                sim[:, i, :] = torch.nn.functional.cosine_similarity(pos_emb[:, i, :].unsqueeze(1), pos_emb_grid_flat.unsqueeze(0), dim=-1)
        elif sim_method == 'l2':
            for i in range(NUM_COORDS):
                sim[:, i, :] = -torch.norm(pos_emb[:, i, :].unsqueeze(1) - pos_emb_grid_flat.unsqueeze(0), dim=-1) # negative l2 distance
        closest_indices = sim.argmax(dim=-1) # shape (B, num_coords)

        # convert the indices to 3D coordinates
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        x_grid, y_grid, z_grid = voxel_size

        x_steps = int((x_max - x_min) / x_grid)
        y_steps = int((y_max - y_min) / y_grid)
        z_steps = int((z_max - z_min) / z_grid)

        # Reshape the closest indices to 3D coordinates
        x_idx = closest_indices // (y_steps * z_steps)  # shape (B, num_coords)
        y_idx = (closest_indices % (y_steps * z_steps)) // z_steps  # shape (B, num_coords)
        z_idx = (closest_indices % (y_steps * z_steps)) % z_steps  # shape (B, num_coords)

        decoded_pos = torch.zeros((B, pos_emb.shape[1], 3), device=pos_emb.device, dtype=torch.float32)  # shape (B, num_coords, 3)
        decoded_pos[:, :, 0] = x_min + x_idx * x_grid
        decoded_pos[:, :, 1] = y_min + y_idx * y_grid
        decoded_pos[:, :, 2] = z_min + z_idx * z_grid
        decoded_pos = decoded_pos.view(B, -1, 3)  # shape (B, num_coords, 3)

        
        # find the 9 voxels around the each decoded position (on xy plane)
        decoded_pos = decoded_pos.view(B, -1, 3)
        decoded_pos_around = decoded_pos.unsqueeze(-2).repeat(1,1, 9, 1) + torch.tensor([ # ignore the z axis for now
            [-x_grid, -y_grid, 0],
            [-x_grid, 0, 0],
            [-x_grid, y_grid, 0],
            [0, -y_grid, 0],
            [0, 0, 0],
            [0, y_grid, 0],
            [x_grid, -y_grid, 0],
            [x_grid, 0, 0],
            [x_grid, y_grid, 0]
            
        ], device=pos_emb.device).view(B, -1, 9, 3) # shape (B, num_coords, 9, 3)

        # Use interpolation regards to the cosine similarity to get the final position
        sim_3d = sim.view(B, -1, pos_emb_grid.shape[0], pos_emb_grid.shape[1], pos_emb_grid.shape[2])  # shape (B, num_coords, x_steps, y_steps, z_steps)
        # use decoded_pos_around as index to get sim for the 9 voxels of each decoded position
        final_pos = torch.zeros_like(decoded_pos, dtype=decoded_pos.dtype) # shape (B, num_coords, 3)
        for coord_idx in range(decoded_pos_around.shape[1]):
            # Get the indices of the 9 voxels around the decoded position
            x_idx = ((decoded_pos_around[:, coord_idx, :, 0] - x_min) / x_grid).long()
            y_idx = ((decoded_pos_around[:, coord_idx, :, 1] - y_min) / y_grid).long()
            z_idx = ((decoded_pos_around[:, coord_idx, :, 2] - z_min) / z_grid).long()
            # remove out of bounds indices
            valid_mask = (x_idx >= 0) & (x_idx < x_steps) & \
                         (y_idx >= 0) & (y_idx < y_steps) & \
                         (z_idx >= 0) & (z_idx < z_steps)
            x_idx = x_idx[valid_mask]
            y_idx = y_idx[valid_mask]
            z_idx = z_idx[valid_mask]
            sim_around = sim_3d[:, coord_idx, x_idx, y_idx, z_idx]

            sim_around = sim_around / self.temperature
            sim_around = torch.nn.functional.softmax(sim_around, dim=-1, )

            # Get the final position by weighted sum of the 9 voxels around the decoded position
            final_pos[:, coord_idx, 0] += (decoded_pos_around[:, coord_idx, :, 0][valid_mask] * sim_around).sum(dim=-1)
            final_pos[:, coord_idx, 1] += (decoded_pos_around[:, coord_idx, :, 1][valid_mask] * sim_around).sum(dim=-1)
            final_pos[:, coord_idx, 2] += (decoded_pos_around[:, coord_idx, :, 2][valid_mask] * sim_around).sum(dim=-1)
        final_pos = final_pos.view(B, -1, 3)

        return decoded_pos, final_pos
    

    def decode_pos_gumbel_softmax(self, pos_emb, pos_emb_grid, pc_range, voxel_size, sim_method='cosine',  force_temperature=None, gumbel_temperature=1e-3):
        """
        Decode the positional encoding to get the original position.
        Instead of decode_pos() function which only use 9 similarities around the decoded position,
        this function uses gumbel softmax to sample the closest position in the grid. (gradients are passed to all positions in the grid)

        :param 
            pos_emb: A tensor of size (B, num_coords, channels)
            pos_emb_grid: A tensor of size (x_steps, y_steps, z_steps, channels) with the positional encoding grid.
            pc_range: The range of the point cloud. e.g. [x_min, y_min, z_min, x_max, y_max, z_max].
            voxel_size: The size of a voxel in grid. e.g. [x_grid, y_grid, z_grid].
        :return: A tensor of size (num_coords, 3) with the original position.
        """

        if sim_method not in ['cosine', 'l2']:
            raise RuntimeError("sim_method must be either 'cosine' or 'l2'!")
        if sim_method == 'l2':
            self.temperature = 0.1 # for l2 loss, use 0.1 as temperature
        elif sim_method == 'cosine':
            self.temperature = 0.001
        if force_temperature is not None:
            self.temperature = force_temperature


        pos_emb_grid = pos_emb_grid.to(pos_emb.device)
        B = pos_emb.shape[0]
        NUM_COORDS = pos_emb.shape[1]
        if len(pos_emb.shape) != 3 or pos_emb.shape[-1] != self.org_channels:
            raise RuntimeError("The input tensor has to be 3d with last dimension of size org_channels!")

        # NOTE: 1. use cosine similarity to find the closest position in the grid
        pos_emb_flat = pos_emb.view(-1, self.org_channels)
        pos_emb_grid_flat = pos_emb_grid.view(-1, self.org_channels)

        # THIS takes too many vram, do it iteratively for every coordinate
        # sim = torch.nn.functional.cosine_similarity(pos_emb_flat.unsqueeze(1), pos_emb_grid_flat.unsqueeze(0), dim=-1) # shape (B, num_coords, x_steps * y_steps * z_steps)
        sim = torch.zeros((B, NUM_COORDS, pos_emb_grid_flat.shape[0]), device=pos_emb.device, dtype=pos_emb.dtype)  # shape (B, num_coords, x_steps * y_steps * z_steps)
        if sim_method == 'cosine':
            for i in range(NUM_COORDS):
                # print('shape of pos_emb_flat[:, i, :].unsqueeze(1)', pos_emb[:, i, :].unsqueeze(1).shape)
                # print('shape of pos_emb_grid_flat.unsqueeze(0)', pos_emb_grid_flat.unsqueeze(0).shape)
                sim[:, i, :] = torch.nn.functional.cosine_similarity(pos_emb[:, i, :].unsqueeze(1), pos_emb_grid_flat.unsqueeze(0), dim=-1)
        elif sim_method == 'l2':
            for i in range(NUM_COORDS):
                # print('shape of pos_emb_flat[:, i, :].unsqueeze(1)', pos_emb[:, i, :].unsqueeze(1).shape)
                # print('shape of pos_emb_grid_flat.unsqueeze(0)', pos_emb_grid_flat.unsqueeze(0).shape)
                sim[:, i, :] = -torch.norm(pos_emb[:, i, :].unsqueeze(1) - pos_emb_grid_flat.unsqueeze(0), dim=-1) # negative l2 distance
        # closest_indices = sim.argmax(dim=-1) # shape (B, num_coords) dtype=torch.long # This does not pass gradients, we can use gumble softmax if we only used decoded
        # use hard gumbel softmax to sample the closest position in the grid
        closest_indices_gumbel = torch.nn.functional.gumbel_softmax(sim, tau=gumbel_temperature, hard=True, dim=-1)# (B, num_coords, x_steps * y_steps * z_steps)
        print(f"Closest indices shape: {closest_indices_gumbel.argmax(dim=-1)}")

        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        x_grid, y_grid, z_grid = voxel_size
        x = torch.arange(x_min, x_max, x_grid, device=self.freq.device, dtype=self.freq.dtype)
        y = torch.arange(y_min, y_max, y_grid, device=self.freq.device, dtype=self.freq.dtype)
        # z = torch.arange(z_min, z_max, z_grid, device=self.freq.device, dtype=self.freq.dtype)
        z = torch.zeros((1), device=self.freq.device, dtype=self.freq.dtype)

        grid_coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1) # shape (x_grids, y_grids, z_grids, 3)

        grid_pos_flatted = grid_coords.reshape(-1, 3).to(closest_indices_gumbel.dtype)  # shape (x_grids * y_grids * z_grids, 3)

        decoded_pos = closest_indices_gumbel @ grid_pos_flatted

        return decoded_pos
    

    def decode_pos_full_grid(self, pos_emb, pos_emb_grid, pc_range, voxel_size, sim_method='cosine',  force_temperature=None):
        """
        Decode the positional encoding to get the original position.
        Instead of decode_pos() function which only use 9 similarities around the decoded position,
        this function uses the weighted sum of the whole grid to get the final position.

        :param 
            pos_emb: A tensor of size (B, num_coords, channels)
            pos_emb_grid: A tensor of size (x_steps, y_steps, z_steps, channels) with the positional encoding grid.
            pc_range: The range of the point cloud. e.g. [x_min, y_min, z_min, x_max, y_max, z_max].
            voxel_size: The size of a voxel in grid. e.g. [x_grid, y_grid, z_grid].
        :return: A tensor of size (num_coords, 3) with the original position.
        """

        if sim_method not in ['cosine', 'l2']:
            raise RuntimeError("sim_method must be either 'cosine' or 'l2'!")
        if sim_method == 'l2':
            self.temperature = 0.1 # for l2 loss, use 0.1 as temperature
        elif sim_method == 'cosine':
            self.temperature = 0.001
        if force_temperature is not None:
            self.temperature = force_temperature


        pos_emb_grid = pos_emb_grid.to(pos_emb.device)
        B = pos_emb.shape[0]
        NUM_COORDS = pos_emb.shape[1]
        if len(pos_emb.shape) != 3 or pos_emb.shape[-1] != self.org_channels:
            raise RuntimeError("The input tensor has to be 3d with last dimension of size org_channels!")

        # NOTE: 1. use cosine similarity to find the closest position in the grid
        X, Y, Z = pos_emb_grid.shape[:3]  # shape (x_steps, y_steps, z_steps)
        pos_emb_grid_flat = pos_emb_grid.view( X*Y*Z, -1 )

        if self.pe_type == 'fone':
            pos_emb = pos_emb[:, :, :self.fone_dim * 3]  # shape (B, num_coords, fone_dim)
            pos_emb_grid_flat = pos_emb_grid_flat[ :, :self.fone_dim * 3]  # shape (x_steps * y_steps * z_steps, fone_dim * 3)

        # THIS takes too many vram, do it iteratively for every coordinate
        # sim = torch.nn.functional.cosine_similarity(pos_emb_flat.unsqueeze(1), pos_emb_grid_flat.unsqueeze(0), dim=-1) # shape (B, num_coords, x_steps * y_steps * z_steps)
        sim = torch.zeros((B, NUM_COORDS, pos_emb_grid_flat.shape[0]), device=pos_emb.device, dtype=pos_emb.dtype)  # shape (B, num_coords, x_steps * y_steps * z_steps)
        if sim_method == 'cosine':
            for i in range(NUM_COORDS):
                sim[:, i, :] = torch.nn.functional.cosine_similarity(pos_emb[:, i, :].unsqueeze(1), pos_emb_grid_flat.unsqueeze(0), dim=-1)
        elif sim_method == 'l2':
            for i in range(NUM_COORDS):
                sim[:, i, :] = -torch.norm(pos_emb[:, i, :].unsqueeze(1) - pos_emb_grid_flat.unsqueeze(0), dim=-1) # negative l2 distance
        sim_softmax = torch.nn.functional.softmax(sim / self.temperature, dim=-1)  # (B, num_coords, x_steps * y_steps * z_steps)

        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        x_grid, y_grid, z_grid = voxel_size
        x = torch.arange(x_min, x_max, x_grid, device=pos_emb.device, dtype=self.freq.dtype)
        y = torch.arange(y_min, y_max, y_grid, device=pos_emb.device, dtype=self.freq.dtype)
        z = torch.zeros((1), device=self.freq.device, dtype=self.freq.dtype)

        grid_coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1) # shape (x_grids, y_grids, z_grids, 3)

        grid_pos_flatted = grid_coords.reshape(-1, 3).to(sim_softmax.dtype).to(pos_emb.device)  # shape (x_grids * y_grids * z_grids, 3)

        decoded_pos = sim_softmax @ grid_pos_flatted


        return decoded_pos


# The following is rope positional encoding implementation for 3D data

class RoPE3D(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        """
        3D RoPE implementation
        :param dim: hidden dim, e.g. 3584
        :param base: frequency base
        """
        super().__init__()
        self.dim = dim
        self.base = base

        axis_dims = [dim // 3] * 3
        remainder = dim % 3
        for i in range(remainder):
            axis_dims[i] += 1  # Distribute the remainder to the first few axes
        # For example, 3584 -> [1195, 1195, 1194]
        self.axis_dims = axis_dims  # list of len 3

        # 2) Each axis must have an even number of positions that can actually be rotated
        # rot_dim is the part that actually participates in RoPE, the tail does not participate
        self.rot_dims = [(d // 2) * 2 for d in self.axis_dims]

        # 3) We build a "maximum required frequency table", and each axis takes a segment from here
        max_rot_dim = max(self.rot_dims)
        # The number of pairs here = max_rot_dim // 2
        rotary_pairs = max_rot_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_pairs).float() / rotary_pairs)
        )  # (P,)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        """
        :param x: (B, N, D)
        :param coords: (B, N, 3)  contains x, y, z coordinates
        :return: (B, N, D) rotated features
        """
        B, N, D = x.shape
        assert D == self.dim, f"Input dim {D} != rope dim {self.dim}"
        assert coords.shape[:2] == (B, N) and coords.shape[-1] == 3, \
            "coords should be (B, N, 3)"

        # Split the hidden dimension into the three axes as we pre-divided
        x_splits = torch.split(x, self.axis_dims, dim=-1)  # list of 3 tensors
        out_splits = []

        for axis_id, (feat_axis, axis_dim, rot_dim) in enumerate(
            zip(x_splits, self.axis_dims, self.rot_dims)
        ):
            # feat_axis: (B, N, axis_dim)

            #  (B, N)
            coord_axis = coords[:, :, axis_id]  # x / y / z

            if rot_dim > 0:
                freq = self.inv_freq[: rot_dim // 2]  # (P,)
                # (B, N, P)
                sinusoid_inp = coord_axis.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
                cos = torch.cos(sinusoid_inp)  # (B, N, P)
                sin = torch.sin(sinusoid_inp)  # (B, N, P)

                # Rotating the rot part
                feat_rot = feat_axis[:, :, :rot_dim]  # (B, N, rot_dim)
                # The remaining tail that does not need rotation
                feat_tail = feat_axis[:, :, rot_dim:]  # (B, N, axis_dim - rot_dim)

                feat_rotated = self.apply_rotary(feat_rot, cos, sin)  # (B, N, rot_dim)

                if feat_tail.numel() > 0:
                    feat_axis_out = torch.cat([feat_rotated, feat_tail], dim=-1)
                else:
                    feat_axis_out = feat_rotated
            else:
                # No rotatable dimensions, return as is
                feat_axis_out = feat_axis

            out_splits.append(feat_axis_out)

        # Concatenate back (B, N, D)
        x_out = torch.cat(out_splits, dim=-1)
        return x_out

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Standard RoPE pairwise rotation
        :param x: (B, N, 2P)  part to be rotated, length must be even
        :param cos: (B, N, P)
        :param sin: (B, N, P)
        :return: (B, N, 2P)
        """
        B, N, D = x.shape
        assert D % 2 == 0, "rotary part must have even dim"
        x1 = x[:, :, : D // 2]
        x2 = x[:, :, D // 2 :]

        # x1,x2: (B,N,P); cos,sin:(B,N,P)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)