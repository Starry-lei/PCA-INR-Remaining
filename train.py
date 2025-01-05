

from dataset.data_loader import PCDataset
from dataset.data_loader import PCDValataset
from dataset.data_loader import CurriculumPCDataset
import argparse
import munch
import datetime
import os
import yaml
import torch
import random
import importlib
import logging
import sys
import torch.optim as optim
import wandb
from tqdm import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import pytorch3d
from pytorch3d import loss


def calc_cd(output, gt):
    cd_l1, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=1, point_reduction='mean')
    # cd_l2, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=2, point_reduction='mean')
    return cd_l1, _


def calc_cd2(output, gt):
    # cd_l1, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=1, point_reduction='mean')
    cd_l2, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=2, point_reduction='mean',batch_reduction=None)
    return cd_l2

output_dir= "./output_dir"
wandb.init(project="PCA_flow_displacement", entity="thesis_lei")




# the input is PCA reduced 3D point cloud and the corresponding projection parameters, the output is the 3D point cloud before PCA reduction
# I would like to learn a residual prediction model to predict the residual between the PCA reduced 3D point cloud and the original 3D point cloud
# the point cloud in the dataset is already normalized by the bounding box, and all the points in the dataset are ordered, whcih means points at the same index are correspondence


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))





def train(args):


    if torch.cuda.is_available():
        print(f"Total GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA GPUs are available.")
    
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    accelerator = Accelerator(
        mixed_precision= 'no', #'fp16',
        gradient_accumulation_steps=1,
        # log_with=args.report_to,
        project_dir=os.path.join(output_dir, "logs"),
        fsdp_plugin=None,
        # even_batches=True,
        kwargs_handlers=[init_handler]
    )
    # LOG.info(accelerator.state)
    num_gpus = accelerator.state.num_processes
    print(f"Number of GPUs being used: {num_gpus}")

    num_epochs = args.epochs

    dataset_train = PCDataset(args, 'train')

    # dataset_train = CurriculumPCDataset(args, set_type='train', phase='easy')



    # dataset_val = PCDataset(args, 'val')
    dataset_val= PCDValataset(args, 'val', global_normalization=dataset_train.global_normalization, mean_shape=dataset_train.normalized_mean_shape_pcd, precomputed_ssm=dataset_train.precomputed_ssm)


    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    
    
    logging.info('Length of train dataset:%d', len(dataloader_train))
    logging.info('Length of validation dataset:%d', len(dataloader_val))

    dataloader_train = accelerator.prepare_data_loader(dataloader_train)
    dataloader_val = accelerator.prepare_data_loader(dataloader_val)

    # exit()
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = args.device

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    

    shape_mask_clusters = dataset_train.mask_clusters
    
    model = model_module.Model(args, shape_mask_clusters=shape_mask_clusters)
    # try transformer model used in partSSM
    model = model.to(device) 

    if hasattr(model_module, 'weights_init'):
        model.apply(model_module.weights_init)



    # Awkward workaround to get gradients from odeint_adjoint to lat_params.
    # lat_params = torch.nn.Parameter(
    #     torch.randn(fullset.n_shapes, args.lat_dims) * 1e-1, requires_grad=True
    # )

    # deformer.add_lat_params(lat_params)
    # deformer.to(device)
    # all_model_params = list(deformer.parameters())

     
    best_val_loss = float('inf')   
    lr = args.lr
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))


    
    optimizer = getattr(optim, args.optimizer)  
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",factor=0.1, patience=10, verbose=True, min_lr=1e-6)

  

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)


    the_pca_mean_shape = torch.from_numpy(np.array(dataset_train.normalized_mean_shape_pcd.points)).float().to(device)
    the_pca_mean_shape = the_pca_mean_shape.unsqueeze(0)
    print("shape of the_pca_mean_shape:", the_pca_mean_shape.shape) # shape of the_pca_mean_shape: torch.Size([1,1024, 3])
    
    # add precomputed PCA and initialize it here
    basis_params_evecs= dataset_train.basis_evecs 
    theta_params_std_dev= dataset_train.theta_std_dev

    

    print("see shape basis_params_evecs:",basis_params_evecs.shape)#  (3072, 64)
    print("see shape theta_params_std_dev:",theta_params_std_dev.shape)# (64, 1)

    basis_params = torch.nn.Parameter(torch.from_numpy(basis_params_evecs).float().to(device), requires_grad=False, )
    std_dev_params= torch.nn.Parameter(torch.from_numpy(theta_params_std_dev).float().to(device=device), requires_grad=False,)
    
    # 还需要优化basis_params和std_dev_params两个先验信息吗？

    model.add_lat_params(basis_params, 'basis_params')
    model.add_lat_params(std_dev_params, 'std_dev_params')
    # model.add_model_params(shape_mask_clusters, 'shape_mask_clusters')

    # 还是Follow flowSSM来做，说得通，同时优化，latent space和shape space !

    # flow space latent space and shape space




    # 回头封装一下上面的东西

    flow_lat_params = torch.nn.Parameter(
    torch.randn(dataset_train.data_length, args.lat_dims, device=device) * 1e-1, requires_grad=True, )
    model.add_lat_params(flow_lat_params, 'flow_lat_params')

    # load params:
    load_params=  False # True
    if load_params:
        path= "latest_model_weights.pth" # latest_model_weights # best_model_weights
        model.load_state_dict(torch.load(path))

    criterion_mse = torch.nn.MSELoss(reduction='none') # hybrid_loss with chamfer_loss # reduction='none': reduction='sum'
    criterion_chamfer= calc_cd2


    bug_handling= "./bug_handling"
    # bug_handling= "./bug_handling_no_c2f"
    loss_scaler= 1 # 1024.0
    global_step = 0

    curriculum_learning_phase= [0,1,2] # easy, medium, hard, 平分数据集，每个phase num_epochs/3 epochs



    for epoch in range(num_epochs):
        
        epoch_idx=epoch+1
        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")

        # print(f"Epoch {epoch}, Phase: {dataset_train.phase}, Active samples: {len(dataset_train)}")
        
        for shape_idx, name, pca_theta, pca_input in tqdm_train_loader:
            pca_mean_shape = the_pca_mean_shape.repeat(pca_input.shape[0], 1, 1)# # torch.Size([4, 1024, 3])
            pca_theta= pca_theta.squeeze(1)

            


            flow_lat= flow_lat_params[shape_idx]

            # print("shape of flow_lat:", flow_lat.shape) # torch.Size([4, 10])
            # exit()


            # print("shape of pca_input:", pca_input.shape) # torch.Size([4, 1024, 3])
            mean_latents= torch.zeros_like(flow_lat)
            # batch together source and target shape for two-way loss training
            source_target_points = torch.cat([pca_mean_shape, pca_input], dim=0)
            target_source_points = torch.cat([pca_input, pca_mean_shape], dim=0)
            # print("see source_target_points shape :", source_target_points.shape)#  torch.Size([8, 1024, 3])
            source_target_latents = torch.cat([mean_latents, flow_lat], dim=0)
            target_source_latents = torch.cat([flow_lat, mean_latents], dim=0)
            g_source_target_latents = torch.cat([mean_latents, pca_theta], dim=0)
            g_target_source_latents = torch.cat([pca_theta, mean_latents], dim=0)



            # print("see target_source_latents shape :", target_source_latents.shape)#  torch.Size([8, 64])
            # gt_deform_abs = torch.mean(torch.norm(pca_mean_shape - pca_input, dim=-1))
            # print("gt_deform_abs change:",gt_deform_abs)

            latent_seq = torch.stack([source_target_latents, target_source_latents], dim=1)
            g_latent_seq = torch.stack([g_source_target_latents, g_target_source_latents], dim=1)
            # print("show latent_seq:",latent_seq.shape)            
            deformed_pts, interpo_points_basis_transformed = model(source_target_points[..., :3], latent_seq, pca_guidance_latents=g_latent_seq, shape_name=name)  # Not set to via_hub.
            # print("see epoch:",epoch_idx)
            # print("see epoch % 100:",epoch_idx % 100)
            main_deformed_pts= deformed_pts[-2*pca_input.shape[0]:,:]

            reg_deformed_pts= deformed_pts[:-2*pca_input.shape[0],:]

            # print("show shape of main_deformed_pts",main_deformed_pts.shape) # 2, 1024, 3
            # print("show shape of reg_deformed_pts",reg_deformed_pts.shape) # 6, 1024, 3
            # exit()

            

            if epoch_idx % 50 == 0:
                # points_transformed_vis= points_transformed[0]
                deformed_pts_vis= main_deformed_pts[0]
                deformed_pts_vis_np= deformed_pts_vis.detach().cpu().numpy()
                # points_transformed_vis_np= points_transformed_vis.detach().cpu().numpy()
                deformed_pts_vis_np_pcd= o3d.geometry.PointCloud()
                points_transformed_vis_np_pcd= o3d.geometry.PointCloud()
                deformed_pts_vis_np_pcd.points=o3d.utility.Vector3dVector(deformed_pts_vis_np)
                # points_transformed_vis_np_pcd.points=o3d.utility.Vector3dVector(points_transformed_vis_np)
                # o3d.visualization.draw_geometries([deformed_pts_vis_np_pcd])
                # o3d.visualization.draw_geometries([points_transformed_vis_np_pcd])
                # save the pcd:
                pcd_path= os.path.join(bug_handling,str(epoch_idx)+"_deformed_pts_vis_np_pcd.ply" )
                # points_transformed_vis_np_pcd_path= os.path.join(bug_handling,str(epoch_idx)+"_points_transformed_vis_np_pcd.ply" )
                o3d.io.write_point_cloud(pcd_path,deformed_pts_vis_np_pcd )
                # o3d.io.write_point_cloud(points_transformed_vis_np_pcd_path,points_transformed_vis_np_pcd )



            # pca_reg_loss= []
            # for idx, regloss in enumerate(interpo_points_basis_transformed):
            #     loss_cal= criterion_chamfer(regloss, g_points_transformed[idx])
            #     # size torch.Size([16])
            #     pca_reg_loss.append(loss_cal)
            # pca_reg_loss_tensor= torch.stack(pca_reg_loss)
            # # print("see pca_reg_loss_tensor:",pca_reg_loss_tensor.shape) #  torch.Size([4, 16])
            # pca_reg_loss= torch.mean(pca_reg_loss_tensor, dim=0)
            # print("see pca_reg_loss:",pca_reg_loss.shape) # torch.Size([16])
      
            

            # print("see pca_reg_loss:",pca_reg_loss)
            cd_l1, _= calc_cd(main_deformed_pts, target_source_points)

            loss_mse = mean_flat(criterion_mse(main_deformed_pts, target_source_points)) 

            pca_reg_loss=0
            if model.coarse_to_fine:
                pca_reg_loss = mean_flat(criterion_mse(reg_deformed_pts,interpo_points_basis_transformed))
                pca_reg_loss= pca_reg_loss.reshape((args.guidance_interpolation+1),-1)
                pca_reg_loss= pca_reg_loss.mean(dim=0)

            # print("show pca_reg_loss: ", pca_reg_loss.shape)
            loss_total = 1000*loss_mse+ 0.5*1000*pca_reg_loss 

            if model.coarse_to_fine:
                pca_reg_loss_mean=pca_reg_loss.mean()
                wandb.log({"pca_reg_loss": pca_reg_loss_mean.item()}, step=global_step)

            loss_mse_mean= loss_mse.mean()
            loss= loss_total.mean()
            # loss = loss_mse
            print("see loss_mse:",loss.item())
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()            
            wandb.log({"train_loss": loss.item()}, step=global_step)
            wandb.log({"cd_l1": cd_l1.item()}, step=global_step)
            wandb.log({"loss_mse_mean": loss_mse_mean.item()}, step=global_step)
           
            # wandb.log({"reg_loss": reg_loss.item()}, step=global_step)
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")
            
        global_step += 1
        torch.save(model.state_dict(), 'latest_model_weights.pth')



    print("Training phase ends, start to validate the model")
    exit()
    model.eval()
    with torch.no_grad():
        val_loss = 0
        num_batches= len(dataloader_val)
        # print("num_batches:", num_batches)# 86
        for shape_idx, pca_recon, pca_theta, pca_input in dataloader_val:

            # loss = creterion(model(pca_recon, pca_rep), pca_input)
            pca_mean_shape = the_pca_mean_shape.repeat(pca_input.shape[0], 1, 1)# # torch.Size([4, 1024, 3])
            pca_theta= pca_theta.squeeze(1)
            mean_latents= torch.zeros_like(pca_theta)
            # batch together source and target shape for two-way loss training
            source_target_points = torch.cat([pca_mean_shape, pca_input], dim=0)
            target_source_points = torch.cat([pca_input, pca_mean_shape], dim=0)
            source_target_latents = torch.cat([mean_latents, pca_theta], dim=0)
            target_source_latents = torch.cat([pca_theta, mean_latents], dim=0)

            latent_seq = torch.stack(
                [source_target_latents, target_source_latents], dim=1
            )

            deformed_pts, points_transformed = model(source_target_points[..., :3], latent_seq)  # Not set to via_hub.
            # cd_l1, loss_cd_l2= calc_cd(deformed_pts, target_source_points)
            # loss_cd= criterion(loss_cd_l2, torch.zeros_like(loss_cd_l2))
            loss_mse= loss_scaler* criterion(deformed_pts, target_source_points)
            # loss = loss_cd+loss_mse
            loss = loss_mse


            val_loss += loss.item()


        val_loss /= num_batches

    wandb.log({
        "val_loss": val_loss,
    }, step=global_step)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')
        wandb.log({"best_val_loss": best_val_loss})

    

            


        




def main():
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)

    arg= parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    print_time = datetime.datetime.now().isoformat()[:19]

    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        # print("checking exp_name:",exp_name)
        log_dir = os.path.dirname(args.load_model)
        # print("checking log_dir:",log_dir)
    else:
        if 'encoder' in args:
            exp_name = args.model_name+'_'+args.encoder
        else:
            exp_name = args.model_name
        exp_name += '_'+print_time.replace(':',"-")
        log_dir = os.path.join(args.work_dir, args.dataset, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("log_dir:",log_dir)
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])
    train(args)


    #test() visualize the result

if __name__ == '__main__':
    # conda activate diffTheta
    # python train.py -c cfgs/config.yaml
    main()