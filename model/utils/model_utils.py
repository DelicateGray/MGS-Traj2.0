import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath


class AggregationLayer(nn.Module):
    def __init__(self, insize, outsize):
        super(AggregationLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(insize, outsize * 2),
            nn.BatchNorm1d(outsize * 2),
            nn.ReLU(),
            nn.Linear(outsize * 2, outsize),
        )

    def forward(self, x, x_mask):
        # [修改点 1] 获取动态维度 BN, L, De
        BN, L, De = x.shape  
        
        # [修改点 2] 使用动态维度进行 reshape，不要写死 128*11*15 和 256
        # 原代码：x = self.mlp(x.reshape(128 * 11 * 15, 256)).reshape(128 * 11, 15, -1)
        x = self.mlp(x.reshape(BN * L, -1)).reshape(BN, L, -1)

        x = repadding(x.clone(), x_mask, torch.min(x))
        
        # [修改点 3] MaxPool 的 kernel_size 应为 L，不要写死 15
        # 原代码：maxpool = nn.MaxPool1d(kernel_size=15)
        maxpool = nn.MaxPool1d(kernel_size=L)
        
        # [修改点 4] repeat 次数应为 L，不要写死 15
        # 原代码：max_feature = maxpool(x.transpose(1, 2)).transpose(1, 2).repeat(1, 15, 1)
        max_feature = maxpool(x.transpose(1, 2)).transpose(1, 2).repeat(1, L, 1)
        
        output = torch.cat((x, max_feature), dim=-1)
        return output


class AggregationBlock(nn.Module):
    def __init__(self, insize, outsize, depth):
        super(AggregationBlock, self).__init__()
        self.agg_net = []
        if depth == 1:
            self.agg_net.append(AggregationLayer(insize, outsize // 2))
        elif depth == 2:
            self.agg_net.append(AggregationLayer(insize, outsize))
            self.agg_net.append(AggregationLayer(2 * outsize, outsize // 2))
        elif depth == 3:
            self.agg_net.append(AggregationLayer(insize, outsize // 2))
            self.agg_net.append(AggregationLayer(outsize, outsize))
            self.agg_net.append(AggregationLayer(2 * outsize, outsize // 2))
        self.agg_net = nn.ModuleList(self.agg_net)

    def forward(self, x, x_mask):
        output = x
        L = output.shape[1] # 动态获取长度 L
        for layer in self.agg_net:
            output = layer(output, x_mask)
            
        # [修改点 5] MaxPool 使用动态 L，不要写死 15
        # 原代码：maxpool = nn.MaxPool1d(15)
        maxpool = nn.MaxPool1d(L)
        
        output = maxpool(output.clone().transpose(1, 2)).squeeze()

        return output


#class AggregationLayer(nn.Module):
#    def __init__(self, insize, outsize):  # (256,128)
#        super(AggregationLayer, self).__init__()
#        self.mlp = nn.Sequential(
#            nn.Linear(insize, outsize * 2),
#            nn.BatchNorm1d(outsize * 2),
#            nn.ReLU(),
#            nn.Linear(outsize * 2, outsize),
#        )
#
#    def forward(self, x, x_mask):  # (N,L,128) (N,L)
#        # print(input.shape, input_mask.shape)
#        BN, L, De = x.shape  # (128*11,15,256)
#        x = self.mlp(x.reshape(BN * L, -1)).reshape(BN, L, -1)
#
#        # (BN,L,outsize)
#        x = repadding(x.clone(), x_mask, torch.min(x))
#        # maxpool = nn.MaxPool1d(kernel_size=L)  # L
#        maxpool = nn.MaxPool1d(kernel_size=L)
#        max_feature = maxpool(x.transpose(1, 2)).transpose(1, 2).repeat(1, L, 1)
#        # (BN,L,outsize)→(BN,outsize,L)→(BN,outsize,1)→(BN,1,outsize)→(BN,L,outsize)
#        output = torch.cat((x, max_feature), dim=-1)
#        # (BN,L,2*outsize)
#        return output
#
#
#class AggregationBlock(nn.Module):
#    def __init__(self, insize, outsize, depth):
#        super(AggregationBlock, self).__init__()
#        self.agg_net = []
#        if depth == 1:
#            self.agg_net.append(AggregationLayer(insize, outsize // 2))
#        elif depth == 2:
#            self.agg_net.append(AggregationLayer(insize, outsize))
#            self.agg_net.append(AggregationLayer(2 * outsize, outsize // 2))
#        elif depth == 3:
#            self.agg_net.append(AggregationLayer(insize, outsize // 2))
#            self.agg_net.append(AggregationLayer(outsize, outsize))
#            self.agg_net.append(AggregationLayer(2 * outsize, outsize // 2))
#        self.agg_net = nn.ModuleList(self.agg_net)
#
#    def forward(self, x, x_mask):
#        output = x
#        L = output.shape[1]
#        for layer in self.agg_net:
#            output = layer(output, x_mask)
#        # maxpool = nn.MaxPool1d(L)
#        maxpool = nn.MaxPool1d(L)
#        output = maxpool(output.clone().transpose(1, 2)).squeeze()
#
#        return output


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, drop_prob=0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.drop_path = DropPath(drop_prob) if self.training else nn.Identity()
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, att_mask=None, key_mask=None):
        # print(query.shape, key.shape, value.shape)
        B, Lq, Lk = query.shape[0], query.shape[1], key.shape[1]
        device = query.device
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if att_mask is not None:
            att_mask = repeat(att_mask, "B Lq -> B 1 Lq Lk", Lk=Lk)
            if key_mask is not None:
                key_mask = repeat(key_mask, "B Lk -> B 1 Lq Lk", Lq=Lq)
                att_mask = att_mask.logical_or(key_mask)
        else:
            if key_mask is not None:
                att_mask = repeat(key_mask, "B Lk -> B 1 Lq Lk", Lq=Lq)
            else:
                att_mask = torch.full(
                    (B, 1, Lq, Lk), fill_value=False, dtype=torch.bool
                ).to(device)
        scores = scores.masked_fill(att_mask.bool(), -1e8)
        weights = self.drop_path(F.softmax(scores, dim=-1))  # (B,head_num,Lq,Lk)
        attn_output = torch.matmul(weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        )
        attn_output = self.out_linear(attn_output)

        return attn_output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, head_num, drop_prob):
        super(TransformerEncoderBlock, self).__init__()
        self.self_att = MultiheadAttention(hidden_size, head_num, drop_prob)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.drop_path_1 = DropPath(drop_prob) if self.training else nn.Identity()
        self.drop_path_2 = DropPath(drop_prob) if self.training else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
            nn.Dropout(drop_prob),
        )

    def forward(self, x, mask):
        temp_1 = self.norm_1(x)
        temp_1 = self.self_att(temp_1, temp_1, temp_1, mask)
        x = x + self.drop_path_1(temp_1)
        temp_2 = self.norm_2(x)
        temp_2 = self.mlp(temp_2)
        x = x + self.drop_path_2(temp_2)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size, head_num, drop_prob):
        super(TransformerDecoderBlock, self).__init__()
        self.self_att = MultiheadAttention(hidden_size, head_num, drop_prob)
        self.cross_att = MultiheadAttention(hidden_size, head_num, drop_prob)

        self.drop_path_1 = DropPath(drop_prob) if self.training else nn.Identity()
        self.drop_path_2 = DropPath(drop_prob) if self.training else nn.Identity()
        self.drop_path_3 = DropPath(drop_prob) if self.training else nn.Identity()

        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.norm_3 = nn.LayerNorm(hidden_size)
        self.norm_4 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
            nn.Dropout(drop_prob),
        )

    def forward(self, x, y, x_mask=None, y_mask=None):
        temp_1 = self.norm_1(x)
        temp_1 = self.self_att(temp_1, temp_1, temp_1, x_mask)
        x = x + self.drop_path_1(temp_1)

        temp_2 = self.norm_2(x)
        y = self.norm_3(y)
        temp_2 = self.cross_att(temp_2, y, y, x_mask, y_mask)
        x = x + self.drop_path_2(temp_2)

        temp_3 = self.norm_4(x)
        temp_3 = self.mlp(temp_3)
        x = x + self.drop_path_3(temp_3)

        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_embeddings, hidden_size):
        super(PositionEmbedding, self).__init__()
        # self.is_absolute = True
        self.embeddings = nn.Embedding(max_embeddings, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[: x.size(-2)]
        # print(x.shape)
        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]
        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, "b h l d -> b l (h d)")
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, "b l (h d) -> b h l d", h=h)
            return x


class TrajMasking:
    def __init__(self, traj_mask, alpha=0.5, beta=0.5):
        self.traj_mask = traj_mask  # (B,N,Th)
        self.B, self.N, self.Th = self.traj_mask.shape
        self.Trar = int(self.Th * alpha)
        self.Tsar = int(self.Th * beta)
        self.device = self.traj_mask.device

    def random_masking(self, traj):
        # 1. 识别有效车辆 (全0为有效，含1为Padding/无效)
        is_valid_agent = torch.all(self.traj_mask == 0, dim=-1) # (B, N)
        is_invalid_agent = ~is_valid_agent # (B, N)
        
        # 2. 生成随机 Mask 索引 (B, N, Trar)
        rand_scores = torch.rand((self.B, self.N, self.Th), device=self.device)
        _, indices = torch.sort(rand_scores, dim=-1)
        mask_indices = indices[..., :self.Trar]
        
        # 3. 构建 rar_traj_mask (输入给 Encoder 的 Mask)
        rar_traj_mask = self.traj_mask.clone()
        random_mask_map = torch.zeros_like(self.traj_mask, dtype=torch.bool)
        random_mask_map.scatter_(2, mask_indices, True)
        rar_traj_mask = rar_traj_mask | random_mask_map
        
        # 4. 构建 rar_target_mask (用于 Loss 计算)
        # 默认全 True (不计算)，被 Mask 的位置设为 False (计算)
        rar_target_mask = torch.ones_like(self.traj_mask, dtype=torch.bool)
        rar_target_mask.scatter_(2, mask_indices, False)
        
        # [关键修复] 对于无效(部分填充)车辆，强制全设为 True (完全不计算 Loss)
        # 这保证了 target[~mask] 的数量严格是 N_valid * Trar
        rar_target_mask.masked_fill_(is_invalid_agent.unsqueeze(-1), True)

        # 5. 构建 rar_rebuild_mask (用于从 Decoder 输出中提取有效车辆)
        # 无效车辆全Mask(1)，有效车辆全保留(0) -> 实际上在 rebuild_model 中是取反使用的
        # 形状需要扩展到 (B, N, Trar)
        rar_rebuild_mask = is_invalid_agent.unsqueeze(-1).expand(-1, -1, self.Trar) 

        # 6. 向量化 Kinematic Prediction
        time_idx = torch.arange(self.Th, device=self.device).view(1, 1, self.Th).expand(self.B, self.N, self.Th)
        dist_matrix = time_idx.unsqueeze(-1) - time_idx.unsqueeze(-2) 
        dist_abs = torch.abs(dist_matrix).float()
        
        # Masked 点距离设为无限大
        col_mask = rar_traj_mask.unsqueeze(-2).expand(-1, -1, self.Th, -1)
        dist_abs = dist_abs.masked_fill(col_mask, 1e9)
        
        delta_t_abs, closest_indices = torch.min(dist_abs, dim=-1)
        delta_t = torch.gather(dist_matrix, -1, closest_indices.unsqueeze(-1)).squeeze(-1) * 0.2

        D = traj.shape[-1]
        closest_indices_expanded = closest_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        ref_states = torch.gather(traj, 2, closest_indices_expanded)
        
        X0, Y0 = ref_states[..., 0], ref_states[..., 1]
        Vx0, Vy0 = ref_states[..., 2], ref_states[..., 3]
        Ax0, Ay0 = ref_states[..., 4], ref_states[..., 5]
        
        sign_dt = torch.sign(delta_t)
        abs_dt = torch.abs(delta_t)
        X1 = X0 + sign_dt * (Vx0 * abs_dt + 0.5 * Ax0 * (abs_dt ** 2))
        Y1 = Y0 + sign_dt * (Vy0 * abs_dt + 0.5 * Ay0 * (abs_dt ** 2))
        
        kinematic_pred_full = torch.stack([X1, Y1], dim=-1)
        
        # 提取被 Mask 的部分
        idx_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        kinematic_pred_masked = torch.gather(kinematic_pred_full, 2, idx_expanded)
        
        # 只保留有效车辆
        kinematic_rebuild = kinematic_pred_masked[is_valid_agent]

        # Meta Index
        batch_ids = torch.arange(self.B, device=self.device).view(self.B, 1).expand(self.B, self.N)
        rar_meta_index = batch_ids[is_valid_agent]

        return (
            rar_traj_mask.to(torch.bool),
            rar_rebuild_mask.to(torch.bool),
            rar_target_mask.to(torch.bool),
            kinematic_rebuild,
            rar_meta_index
        )

    def social_masking(self):
        sar_traj_mask = self.traj_mask.clone()
        sar_target_mask = self.traj_mask.clone()
        
        indices = torch.arange(self.Th, device=self.device)
        mask_condition = indices >= (self.Th - self.Tsar)
        mask_map = mask_condition.view(1, 1, -1).expand(self.B, self.N, self.Th)
        
        is_valid_agent = torch.all(self.traj_mask == 0, dim=-1) # (B, N)
        is_invalid_agent = ~is_valid_agent
        
        final_mask_map = mask_map & is_valid_agent.unsqueeze(-1)
        sar_traj_mask = sar_traj_mask | final_mask_map
        
        sar_target_mask.fill_(True)
        sar_target_mask.masked_fill_(final_mask_map, False)
        
        # [关键修复] 同样强制无效车辆不计算 Target
        sar_target_mask.masked_fill_(is_invalid_agent.unsqueeze(-1), True)
        
        sar_rebuild_mask = is_invalid_agent.unsqueeze(-1).expand(-1, -1, self.Tsar)

        batch_ids = torch.arange(self.B, device=self.device).view(self.B, 1).expand(self.B, self.N)
        sar_meta_index = batch_ids[is_valid_agent]

        return (
            sar_traj_mask.to(torch.bool),
            sar_rebuild_mask.to(torch.bool),
            sar_target_mask.to(torch.bool),
            sar_meta_index
        )

#class TrajMasking:
#    def __init__(self, traj_mask, alpha=0.5, beta=0.5):
#        self.traj_mask = traj_mask  # (B,N,Th)
#        self.B, self.N, self.Th = self.traj_mask.shape
#        self.Trar = int(self.Th * alpha)
#        self.Tsar = int(self.Th * beta)
#        self.ValidAgentIndex = torch.all(self.traj_mask == 0, dim=-1)  # (B,N)
#        self.device = self.traj_mask.device
#
#    def random_masking(self, traj):
#        rar_traj_mask, rar_target_mask, rar_rebuild_mask, kinematic_rebuild, rar_meta_index = (
#            copy.deepcopy(self.traj_mask),
#            copy.deepcopy(self.traj_mask),
#            torch.zeros((self.B, self.N, self.Trar), dtype=torch.int).to(self.device),
#            [], []
#        )
#        for i in range(self.B):
#            for j in range(self.N):
#                if self.ValidAgentIndex[i, j] == 0:
#                    rar_rebuild_mask[i, j, :] = 1
#                    rar_target_mask[i, j, :] = 1
#                else:
#                    agt_mask_ij = rar_traj_mask[i, j, :]
#                    tgt_mask_ij = rar_target_mask[i, j, :]
#                    traj_ij = traj[i, j, :, :]
#                    if 0 < self.Trar < self.Th:
#                        index = torch.randperm(self.Th)
#                        mask_index = index[: self.Trar]
#                        unmask_index = index[self.Trar:]
#                        kinematic_rebuild.append(
#                            self.kinematic_prediction(traj_ij, mask_index, unmask_index).unsqueeze(0))
#                        agt_mask_ij[unmask_index] = 0
#                        agt_mask_ij[mask_index] = 1
#                        tgt_mask_ij[unmask_index] = 1
#                        tgt_mask_ij[mask_index] = 0
#                        rar_meta_index.append(i)
#                        rar_traj_mask[i, j, :], rar_target_mask[i, j, :] = agt_mask_ij, tgt_mask_ij
#        return (
#            rar_traj_mask.to(torch.bool),
#            rar_rebuild_mask.to(torch.bool),
#            rar_target_mask.to(torch.bool),
#            torch.cat(kinematic_rebuild, dim=0).to(self.device),
#            torch.tensor(rar_meta_index).to(self.device)
#        )
#
#    def kinematic_prediction(self, traj, masked_index, unmasked_index):
#        kinematic_index = []
#        deltaT = []
#        for idx in masked_index:
#            delta = torch.Tensor([idx]) - unmasked_index
#            uidx = torch.argmin(np.abs(delta))
#            kinematic_index.append(unmasked_index[uidx])
#            deltaT.append(delta[uidx] * 0.2)
#        kinematic_index = torch.tensor(kinematic_index)
#        deltaT = torch.tensor(deltaT).to(self.device)
#        X0 = traj[kinematic_index, 0]
#        Y0 = traj[kinematic_index, 1]
#        Vx0 = traj[kinematic_index, 2]
#        Vy0 = traj[kinematic_index, 3]
#        Ax0 = traj[kinematic_index, 4]
#        Ay0 = traj[kinematic_index, 5]
#        X1 = X0 + torch.sign(deltaT) * (Vx0 * deltaT.abs() + 0.5 * Ax0 * deltaT.square())
#        Y1 = Y0 + torch.sign(deltaT) * (Vy0 * deltaT.abs() + 0.5 * Ay0 * deltaT.square())
#
#        return torch.cat((X1.unsqueeze(1), Y1.unsqueeze(1)), dim=1)
#
#    def social_masking(self):
#        sar_traj_mask, sar_target_mask, sar_rebuild_mask, sar_meta_index = (
#            copy.deepcopy(self.traj_mask),
#            copy.deepcopy(self.traj_mask),
#            torch.zeros((self.B, self.N, self.Tsar), dtype=torch.int).to(self.device), []
#        )
#        for i in range(self.B):
#            for j in range(self.N):
#                if self.ValidAgentIndex[i, j] == 0:
#                    sar_rebuild_mask[i, j, :] = 1
#                    sar_target_mask[i, j, :] = 1
#                else:
#                    agt_mask_ij = sar_traj_mask[i, j, :]
#                    tgt_mask_ij = sar_target_mask[i, j, :]
#                    if 0 < self.Tsar < self.Th:
#                        index = torch.arange(self.Th)
#                        mask_index = index[-self.Tsar:]
#                        unmask_index = index[:self.Th - self.Tsar]
#                        agt_mask_ij[unmask_index] = 0
#                        agt_mask_ij[mask_index] = 1
#                        tgt_mask_ij[unmask_index] = 1
#                        tgt_mask_ij[mask_index] = 0
#                        sar_traj_mask[i, j, :] = agt_mask_ij
#                        sar_target_mask[i, j, :] = tgt_mask_ij
#                        sar_meta_index.append(i)
#
#        return (
#            sar_traj_mask.to(torch.bool),
#            sar_rebuild_mask.to(torch.bool),
#            sar_target_mask.to(torch.bool),
#            torch.tensor(sar_meta_index).to(self.device)
#        )


# class LaneMasking:
#     def __init__(self, lane_mask, gamma=0.5):
#         # (B,N,Th) (B,N,Th+1)
#         self.lane_mask = lane_mask
#         self.B, self.M, self.L = self.lane_mask.shape
#         self.Lrlr = int(self.L * gamma)
#         self.ValidLaneIndex = torch.all(self.lane_mask == 0, dim=-1)  # (B,N)
#         self.device = self.lane_mask.device
#
#     def random_masking(self):
#         rlr_lane_mask, rlr_target_mask, rlr_pred_mask, rlr_meta_index = (
#             copy.deepcopy(self.lane_mask),
#             torch.zeros((self.B, self.M, config.L), dtype=torch.int).to(self.device),
#             torch.zeros((self.B, self.M, self.Lrlr), dtype=torch.int).to(self.device), []
#         )
#
#         for i in range(self.B):
#             for j in range(self.M):
#                 if self.ValidLaneIndex[i, j] == 0:
#                     rlr_pred_mask[i, j, :] = 1
#                     rlr_target_mask[i, j, :] = 1
#                 else:
#                     lane_mask_ij = rlr_lane_mask[i, j, :]
#                     lane_target_ij = rlr_target_mask[i, j, :]
#                     if 0 < self.Lrlr < self.L:
#                         index = torch.randperm(self.L)
#                         mask_index = index[: self.Lrlr]
#                         unmask_index = index[self.Lrlr:]
#                         lane_mask_ij[unmask_index] = 0
#                         lane_mask_ij[mask_index] = 1
#                         lane_target_ij[unmask_index] = 1
#                         lane_target_ij[mask_index] = 0
#                         # lane_target_ij[mask_index[-1]+1] = 0
#                         print(torch.sum(lane_target_ij==0),torch.sum(lane_target_ij==1))
#                         rlr_lane_mask[i, j, :], rlr_target_mask[i, j, :] = (
#                             lane_mask_ij,
#                             lane_target_ij,
#                         )
#                         rlr_meta_index.append(i)
#
#         return (
#             rlr_lane_mask.to(torch.bool),
#             rlr_pred_mask.to(torch.bool),
#             rlr_target_mask.to(torch.bool),
#             torch.tensor(rlr_meta_index).to(self.device)
#         )

def repadding(feature, feature_mask, padding_value):
    # print(feature.shape, feature_mask.shape, padding_value)
    feature_clone = feature.clone()
    feature_clone[feature_mask.unsqueeze(-1).repeat(1, 1, feature_clone.shape[2]).bool()] = padding_value
    padded_feature = feature_clone

    return padded_feature


def safe_softmax(x, dim=-1, epsilon=1e-8):
    # 对输入进行平移，即减去输入中的最大值
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    shifted_x = x - max_val

    # 计算平移后的 softmax，加入一个小的正值 epsilon 到分母中
    exp_x = torch.exp(shifted_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax_output = exp_x / (sum_exp_x + epsilon)

    return softmax_output
