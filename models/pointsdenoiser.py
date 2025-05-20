# Proposed model - Point2SSM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.ae import PointNet_encoder
from models.dgcnn import DGCNN_encoder
from models.src.model import Encode2Points
from torch_cluster import knn
import torch_cluster
# import xformers.ops
import os
import time
import sys
from models.src.network.utils import normalize_3d_coordinate, ResnetBlockFC, normalize_coordinate
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
from pointnet2_utils import grouping_operation
from utils.model_utils import calc_cd

criterion_mse = torch.nn.MSELoss(reduction='sum') # reduce=None #  with chamfer_loss reduction='sum': reduction='sum'
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

from models.Transformer_utils import *




class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        # print("show shape of g_feature:",g_feature.shape)
        token_feature = rec_feature
        # print("show shape of token_feature:",token_feature.shape)


        #  rebuild_feature = torch.cat([
        #     global_feature.unsqueeze(-2).expand(-1, M, -1),
        #     q,
        #     coarse_point_cloud], dim=-1)  # B M 1027 + C
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        # print("show shape of patch_feature:",patch_feature.shape)

        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.train_loss = args.loss
        self.device = args.device
        self.alpha = args.alpha

        self.encoder_name = args.model_name

        # print("ssss self.latent_dim:",self.latent_dim)
        if args.encoder == 'dgcnn':
            print("ssss use dgcnn:",self.latent_dim)
            self.encoder = DGCNN_encoder(self.latent_dim)
        elif args.encoder == 'local_pool_pointnet':
            self.encoder = Encode2Points(args)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        self.attn_module = Attention_Module(self.latent_dim, self.num_output)

        self.denoiser_module= SimpleRebuildFCLayer((self.latent_dim+3)*2,  step=1)


        # rebuild_feature = torch.cat([
        #     global_feature.unsqueeze(-2).expand(-1, M, -1),
        #     q,
        #     coarse_point_cloud], dim=-1)  # B M 1027 + C

        

        # self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)

    # all combos within batch
    def get_neighbor_loss(self, pred, k):
        edge_index = [knn(pred[i], pred[i], k, ) for i in range(pred.shape[0])]
        neigh_idxs = torch.stack([edge_index[i][1].reshape(pred.shape[1], -1) for i in range(pred.shape[0])])
        batch_size = pred.shape[0]
        loss, count = 0, 0

        for source_index in range(batch_size):
            for target_index in range(batch_size):
                if source_index != target_index:
                    count += 1
                    loss += self.neighbor_loss_helper(pred[source_index].unsqueeze(0),
                                                      neigh_idxs[source_index].unsqueeze(0),
                                                      pred[target_index].unsqueeze(0), k)
        return loss / count

    def neighbor_loss_helper(self, source, source_neighs, target, k):
        source_grouped = grouping_operation(source.transpose(1, 2).contiguous(), source_neighs.int()).permute(0, 2, 3,
                                                                                                              1)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source,
                                                                    2)  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff ** 2, dim=-1)

        target_cr_grouped = grouping_operation(target.transpose(1, 2).contiguous(), source_neighs.int()).permute(0, 2,
                                                                                                                 3, 1)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target,
                                                                          2)  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square / GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.sum(neighbor_loss_per_neigh)

        return neighbor_loss
    


    # def forward(self, x, gt=None, is_training=True):
    #     # print("show shape of x:",x.shape)  # torch.Size([5, 2048, 3])
    #     z, features = self.encoder(x)
    #     residuals = self.denoiser_module(x, features) # [B, N, 3]
    #     pred =  x + residuals
    #     if is_training:            
    #         cd_p, cd_t = calc_cd(pred, gt)
    #         recon_loss = cd_t
    #         loss= mean_flat(recon_loss) #+ self.alpha * mean_flat(neigh_loss)
    #         return pred, loss
    #     else:
    #         return pred




    def forward(self, x, gt=None, is_training=True):

        # print("show shape of x:",x.shape)  # torch.Size([5, 2048, 3])
        global_feature, features = self.encoder(x)
        features = features.transpose(2, 1)  # [B, N, latent_dim]
        # print("show shape of transposed features:", features.shape) # [20, 128, 2048]
        # print("show shape of global_feature:",global_feature.shape)  #torch.Size([20, 128])
        # exit()
        residuals = self.attn_module(features, coords=x, global_feature=global_feature) # [B, N, 3]
        # residuals = self.denoiser_module(x, features) # [B, N, 3]

        # self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        # add a folder net
        # print("show shape of gt:", gt.shape)
        pred =  x + residuals


        pred = torch.cat([pred,global_feature.unsqueeze(1).expand(-1, pred.shape[1], -1)], dim=-1)
        # print("show shape of pred:", pred.shape) # b, 2048, 131
        # output= 
        pred=x+self.denoiser_module(pred).squeeze(2)

        # print("show shape of gt:", gt.shape)# b, 8192,3
        # print("show shape of final out:", out.shape) # b, 2048,3
        # exit()


        if is_training:            
            cd_p, cd_t = calc_cd(pred.float(), gt.float())
            recon_loss = cd_t
            loss= mean_flat(recon_loss) #+ self.alpha * mean_flat(neigh_loss)
            return pred, loss
        else:
            return pred
        

    # def forward(self, x, gt=None, is_training=True):
      
    #     out = self.encoder(x)
    #     points, normals = out
    #     pred= points-normals 
    #     # print("show shape of points:",points.shape)
    #     # density num_offset=7
    #     #show shape of points: torch.Size([16, 14336, 3])
    #     # print("show shape of normals:",normals.shape)
    #     # density num_offset=7
    #     # show shape of normals: torch.Size([16, 14336, 3])
    #     # density num_offset=1
    #     # show shape of normals: torch.Size([16, 2048, 3])
    #     if self.encoder_name == 'dgcnn':
    #         features = features.transpose(2, 1)

    #     # prob_map = self.attn_module(features)
    #     # pred = torch.sum(prob_map[:, :, :, None] * x[:, None, :, :], dim=2)

    #     if is_training:
    #         # cd_p, cd_t = criterion_mse(pred, gt)
    #         # loss_mse = criterion_mse(pred, gt)
    #         cd_p, cd_t = calc_cd(pred, gt)
    #         recon_loss = cd_t
    #         # recon_loss=  mean_flat(loss_mse)            
    #         # neigh_loss = self.get_neighbor_loss(pred, 10)
    #         loss= mean_flat(recon_loss) #+ self.alpha * mean_flat(neigh_loss)
    #         return pred, loss
    #     else:
   
    #         return pred
        



    def get_prob_map(self, x):
        z, features = self.encoder(x)

        if self.encoder_name == 'dgcnn':
            features = features.transpose(2, 1)

        prob_map = self.attn_module(features)

        pred = torch.sum(prob_map[:, :, :, None] * x[:, None, :, :], dim=2)

        return prob_map, pred


# class Attention_Module(nn.Module):
#     def __init__(self, latent_dim, num_output):
#         super(Attention_Module, self).__init__()
#         self.num_output = num_output
#         self.latent_dim = latent_dim
#         print("++++++++++++++++++++++++++++++++++++++++++checking latent_dim, num_output:", latent_dim,
#               num_output)  # 128 2048
#         time_rec_1 = time.time()
#         self.sa1 = cross_transformer(self.latent_dim, self.num_output)
#         self.sa2 = cross_transformer(self.num_output, self.num_output)
#         self.sa3 = cross_transformer(self.num_output, self.num_output)
#         self.softmax = nn.Softmax(dim=2)
#         print("cross atten using time:", str(time.time() - time_rec_1)[:4])

#     def forward(self, x):
#         x = self.sa1(x, x)
#         x = self.sa2(x, x)
#         x = self.sa3(x, x)
#         prob_map = self.softmax(x)
#         return prob_map


class ImNet(nn.Module): # develop a transformation based IMNet?
    """ImNet layer pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=128,
        out_features=3,
        nf=16,
        nonlinearity="leakyrelu",
    ):
        """Initialization.

        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = nn.LeakyReLU()
        self.fc0 = nn.Linear(self.dimz, nf * 16)
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        # A coord-condition MLP
        """Forward method. 
        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)

        x_tmp = self.activ(self.fc4(x_tmp))

        x_tmp = self.fc5(x_tmp)
        return x_tmp



class Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output, n_blocks= 3):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim
        self.coor_dim=3


        self.n_blocks=n_blocks

        self.use_global_feature =True


        # self.sa1 = cross_transformer(self.latent_dim+self.coor_dim,self.num_output)
        # self.sa2 = cross_transformer(self.num_output+self.coor_dim,self.num_output)
        # self.sa3 = cross_transformer(self.num_output+self.coor_dim,self.num_output)
        
        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        # self.sa3 = cross_transformer(self.num_output,self.num_output)


        if not self.use_global_feature:
            self.coord_cond_mlp = ImNet(dim=3,in_features=self.num_output, out_features=3, nf=32)
        else:
            self.coord_cond_mlp = ImNet(dim=3,in_features=self.num_output+self.latent_dim, out_features=3, nf=32)





       
        # Final projection to residuals [B, N, latent_dim] -> [B, N, 3]
        # self.residual_proj = nn.Linear(num_output, 3)


        for m in self.coord_cond_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                # nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x, coords=None,global_feature= None):


        # print("show shape of x before sa1:",x.shape)
        # torch.Size([5, 128+3, 2048])
        # print("show shape of x before sa1:",x.shape)
        # torch.Size([5, 128+3, 2048])


        # coords_in = coords.transpose(2, 1)

        # x= torch.cat([coords_in, x], dim=1)
        x = self.sa1(x,x)

        # x= torch.cat([coords_in, x], dim=1)
        x = self.sa2(x,x)

        # x= torch.cat([coords_in, x], dim=1)
        # x = self.sa3(x,x)


        x = x.permute(0,2,1)
        # print("show shape of x:",x.shape) # 5, 2048, 256
        # print("show shape of coords:",coords.shape)# 5, 2048, 3

        if not self.use_global_feature:
            net_input = torch.cat([coords, x], dim=-1) # add global feature here
            out = self.coord_cond_mlp(net_input)
        else:
            # global_feature 20, 128
            # rebuild_feature = torch.cat([global_feature.unsqueeze(1).expand(-1, M, -1),q,coarse_point_cloud], dim=-1)  # B M 1027 + C
            net_input = torch.cat([coords,x,global_feature.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1) # add global feature here， 20, 2048, 387
            # print("show shape of net_input:",net_input.shape)
            # exit()
            out = self.coord_cond_mlp(net_input)


        # out = self.residual_proj(nn.LeakyReLU(net))
        # print("show shape of out:",out.shape) # e([5, 2048, 3])  
        # exit()

        return out
    



class Residual_Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output, n_blocks= 3):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim

        self.n_blocks=n_blocks
        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        # self.sa3 = cross_transformer(self.num_output,self.num_output)
        # Final projection to residuals [B, N, latent_dim] -> [B, N, 3]
        self.residual_proj = nn.Linear(num_output, 3)
    def forward(self, x, coords=None):

        x = self.sa1(x,x)
        x = self.sa2(x,x)
        # x = self.sa3(x,x)
        

        # x = x.permute(0,2,1)
        # print("show shape of x:",x.shape) # 5, 2048, 256
        # print("show shape of coords:",coords.shape)# 5, 2048, 3
        # net_input = torch.cat([coords, x], dim=-1)
        # out = self.coord_cond_mlp(net_input)

        out = self.residual_proj(x)
        # print("show shape of out:",out.shape) # e([5, 2048, 3])  

        return out
    

class denoiser_module(nn.Module):
    def __init__(self, dim=3, c_dim=128, out_dim=3, hidden_size=128, n_blocks=5, leaky=False):
        super(denoiser_module, self).__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.in_dim = dim 

        self.fc_in = nn.Linear(self.in_dim+self.c_dim, hidden_size)
        

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)


        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size, activation=self.actvn) for _ in range(n_blocks)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(n_blocks)
        ])


        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.activation = self.actvn
    def forward(self, coords, global_feature=None):

        B, N, _ = coords.shape

        net_input = torch.cat([coords,global_feature.unsqueeze(1).expand(-1, coords.shape[1], -1)], dim=-1)


        net = self.fc_in(net_input)          # => (B*N, hidden_size)
        net = self.activation(net)


        for block in self.blocks:
            net = block(net)



        out = self.fc_out(self.activation(net))  # => (B*N, out_dim)

        residuals = out  # [B, N, 3]
        return residuals




# PointAttN: You Only Need Attention for Point Cloud Completion
# https://github.com/ohhhyeahhh/PointAttN
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=8, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # self.cross_attn = MultiHeadCrossAttention(d_model_out, nhead, attn_drop=dropout)
        # self.multihead_attn1 = FlashAttentionCrossAttention(d_model_out, nhead, dropout=dropout)
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

    # transformer
    def forward(self, src1, src2, if_act=False):

        # print("\n new**start:**********************************see shape of src1:",src1.shape)
        #  torch.Size([5, 128, 2048])  

        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        # print("\n **start:**********************************see shape of src1:",src1.shape)
        # # [5, 256, 2048]
        # print("****************************************see shape of src2:",src2.shape)
        # # [5, 256, 2048]

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]
        
        # print("********************************************************see shape of src12:",src12.shape)
        # #  torch.Size([2048, 5, 256])
        # print("********************************************************see shape of src1:",src1.shape)
        # #  torch.Size([2048, 5, 256])
        # print("********************************************************see shape of src2:",src2.shape)
        # #  torch.Size([2048, 5, 256])

        # src12 = self.cross_attn(x=src1, cond=src2)
        # print("********************************************************see shape of src12:",src12.shape)
        # cross_attn: see shape of src12: torch.Size([2048, 16, 2048]
        # multihead_attn1 see shape of src12: torch.Size([2048, 16, 2048])
        

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        # print("************************************see shape of src1:",src1.shape)
        # 2048, 5, 256])

        # exit()

        src1 = src1.permute(1, 2, 0)

        return src1


