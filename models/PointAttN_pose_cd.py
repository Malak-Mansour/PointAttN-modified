# Copyright (c) Zheng Dang (zheng.dang@epfl.ch)
# Please cite the following paper if you use any part of the code.
# [-] Zheng Dang, Lizhou Wang, Yu Guo, Mathieu Salzmann, Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World, ECCV2022

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.utils.data

import numpy as np
from utils.model_utils import *
from utils.hypercd_utils.loss_utils import get_loss1


# from utils.mm3d_pn2 import furthest_point_sample, gather_points
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

from utils.transformer import Transformer




class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)
        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse,feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)
        # Pool y3 to shape (batch_size, 2k, 1)
        y_g = F.adaptive_max_pool1d(y3.permute(0, 2, 1), 1).view(batch_size, -1).unsqueeze(-1)
        y_up = y.repeat(1,1,self.ratio)
        y_cat = torch.cat([y3,y_up],dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

        return x, y3, y_g

# PointAttN

class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)

    def forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1,x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()
        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        
        return x_g, fine
 


class MLPHead(nn.Module):
    def __init__(self, emb_dims):
        super(MLPHead, self).__init__()
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation

class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.emb_dims = 64
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        
        
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        # from utils.commons import save_data
        # save_data("test.pth", [src[1].cpu().numpy(), src_corr[1].cpu().numpy()])

        src_centered = src - src.mean(dim=2, keepdim=True)  # [B, 3, N]

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)  # [B, 3, M]

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)
    
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        else:
            ValueError('dataset is not exist')

        emb_dims = 512
        self.pointer = Transformer()
        self.pose_head = MLPHead(emb_dims=emb_dims)
        self.proj_fine = nn.Linear(step1 * emb_dims, emb_dims)
        self.proj_fine1 = nn.Linear(step1 * step2 * emb_dims, 512)
       
        
        self.encoder = PCT_encoder()
        self.refine = PCT_refine(ratio=step1)
        self.refine1 = PCT_refine(ratio=step2)

    # cd loss]
    @staticmethod
    def sample_points(x, num_points):
        """
        Sample points from the input tensor x.
        :param x: Input tensor of shape (B, C, N)
        :param num_points: Number of points to sample
        :return: Sampled points of shape (B, C, num_points)
        """
        indices = furthest_point_sample(x.transpose(1, 2).contiguous(), num_points)
        sampled_points = gather_points(x.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
        return sampled_points
        
    def forward(self, src, tgt=None, models=None, is_training=True):

        assert not torch.isnan(src).any(), f"src contains NaN values: {torch.isnan(src).any()}"
        assert not torch.isnan(tgt).any(), f"tgt contains NaN values: {torch.isnan(tgt).any()}"
        assert not torch.isnan(models).any(), f"models contains NaN values: {torch.isnan(models).any()}"

    
        src_embeddings, coarse = self.encoder(src)
        tgt_sampled = self.sample_points(tgt, 2048).transpose(1, 2).contiguous()
        tgt_embeddings, _ = self.encoder(tgt_sampled)
        # print(f"src_embeddings shape: {src_embeddings.shape}, tgt_embeddings shape: {tgt_embeddings.shape}")
        assert not torch.isnan(src_embeddings).any(), f"src_embedding contains NaN values: {torch.isnan(src_embeddings).any()}"
       


        new_x = torch.cat([src,coarse],dim=2)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        fine, feat_fine, fine_embeddings = self.refine(None, new_x, src_embeddings)
        fine1, feat_fine1, fine1_embeddings = self.refine1(feat_fine, fine, src_embeddings)
        

        
        fine_embeddings = self.proj_fine(fine_embeddings.reshape(fine_embeddings.shape[0], -1)).unsqueeze(-1)
        fine1_embeddings = self.proj_fine1(fine1_embeddings.reshape(fine1_embeddings.shape[0], -1)).unsqueeze(-1)
        

        coarse = coarse.transpose(1, 2).contiguous()
        tgt_sampled = self.sample_points(tgt, coarse.shape[1])

        fine = fine.transpose(1, 2).contiguous()
        tgt_fine = self.sample_points(tgt, fine.shape[1])
        
 
        # indices1 = furthest_point_sample(fine1.transpose(1, 2).contiguous(), gt.shape[1])
        
        fine1 = fine1.transpose(1, 2).contiguous()
        
       

        Rs_pred, ts_pred = self.pose_head(src_embeddings, tgt_embeddings)
        Rs_pred_fine, ts_pred_fine = self.pose_head(fine_embeddings, tgt_embeddings)
        Rs_pred_fine1, ts_pred_fine1 = self.pose_head(fine1_embeddings, tgt_embeddings)
        
        
        models_transformed = torch.matmul(models, Rs_pred.transpose(1, 2)) + ts_pred.unsqueeze(1)
        models_transformed_fine = torch.matmul(models, Rs_pred_fine.transpose(1, 2)) + ts_pred_fine.unsqueeze(1)
        models_transformed_fine1 = torch.matmul(models, Rs_pred_fine1.transpose(1, 2)) + ts_pred_fine1.unsqueeze(1)
        
        
        # print(f"gt Shape {gt_coarse.shape, gt_fine.shape}, Fine shape: {fine.shape}, Fine1 shape: {fine1.shape}, Coarse shape: {coarse.shape}")

        #   PARAMETER: CD, HCD
        enable_hypercd = False

        if is_training:

            if enable_hypercd:
                pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
                # Call get_loss1 with appropriate parameters
                loss_all, losses = get_loss1(pcds_pred, src.transpose(1, 2).contiguous(), tgt, sqrt=True)
                # Unpack individual losses for logging or other purposes
                cdc, cd1, cd_hyp, partial_matching = losses
                loss2 = cd_hyp
                # Add penalty loss to the total loss
                total_train_loss = loss_all 
                
                # For compatibility with the return statement
                
            else:
                            
                loss4 = calc_cd(models_transformed, tgt)[0].mean() 
                + calc_cd(models_transformed_fine, tgt)[0].mean() 
                + calc_cd(models_transformed_fine1, tgt)[0].mean()
                
                # Check for NaNs in all relevant tensors to trace where it starts
                if torch.isnan(loss4).any():
                    print("loss4 is NaN. Debugging intermediate tensors:")
                    
                loss3, _ = calc_cd(fine1, tgt)
                loss2, _ = calc_cd(fine, tgt_fine)
                loss1, _ = calc_cd(coarse, tgt_sampled)
                # print(f"loss1: {loss1.mean()}, loss2: {loss2.mean()}, loss3: {loss3.mean()}, loss4: {loss4.mean()}")
                total_train_loss = loss1.mean() + loss2.mean() + loss3.mean() + loss4

            return fine, loss2, total_train_loss
            

        else:
            if tgt is not None:

            # Call get_loss1 with appropriate parameters
                if enable_hypercd:

                    pcds_pred = [coarse, fine, fine1, fine1]  # Using fine1 twice as P3 (finest resolution)
                    loss_all, losses = get_loss1(pcds_pred, src.transpose(1, 2).contiguous(), tgt, sqrt=True)
                    cdc, cd1, cd_hyp, partial_matching = losses
                else: # set hyper_cd to 0 if it was not enabled 
                    cd_hyp = torch.tensor(0.0)


                cd_p, cd_t, f1 = calc_cd(fine1, tgt, calc_f1=True)
                cd_p_coarse, cd_t_coarse = calc_cd(coarse, tgt_sampled)
                cd_p_models, cd_t_models = calc_cd(models_transformed_fine1, tgt)

                return {
                    'out1': coarse, 'out2': fine1, "models_transformed": models_transformed, 'Rs_pred': Rs_pred, 'ts_pred': ts_pred, 
                    'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t, 'cd_p_models': cd_p_models, 'cd_t_models': cd_t_models
                    }
            else:
                return {'out1': coarse, 'out2': fine1, "models_transformed": models_transformed, 'Rs_pred': Rs_pred, 'ts_pred': ts_pred}
            
            
            
