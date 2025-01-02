

from dataset.data_loader import PCDataset
from dataset.data_loader import PCDValataset
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

import pytorch3d
from pytorch3d import loss


def calc_cd(output, gt):
    cd_l1, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=1, point_reduction='sum')
    cd_l2, _ = pytorch3d.loss.chamfer_distance(output, gt, norm=2, batch_reduction=None, point_reduction='sum')
    return cd_l1, cd_l2

output_dir= "./output_dir"
wandb.init(project="PCA_flow_displacement", entity="thesis_lei")




# the input is PCA reduced 3D point cloud and the corresponding projection parameters, the output is the 3D point cloud before PCA reduction
# I would like to learn a residual prediction model to predict the residual between the PCA reduced 3D point cloud and the original 3D point cloud
# the point cloud in the dataset is already normalized by the bounding box, and all the points in the dataset are ordered, whcih means points at the same index are correspondence

def log_point_clouds_grid( point_clouds, n_rows=2, n_cols=4):
        """
        Create a 4x6 grid of interactive 3D point clouds
        Args:
            point_clouds: list of point clouds, each with shape (4096, 3)
        """
        # Create HTML table for 4x6 grid layout
        html = "<table style='border-spacing: 10px;'>"
        
        for i in range(n_rows):  # 4 rows
            html += "<tr>"
            for j in range(n_cols):  # 6 columns
                idx = i * n_cols + j
                if idx < len(point_clouds):
                    html += "<td style='border: 1px solid gray; padding: 5px;'>"
                    # Log each point cloud directly
                    wandb.log({f"shape_{idx}": wandb.Object3D(point_clouds[idx])})
                    html += f"<div style='width: 250px; height: 250px;'></div>"
                    html += f"<div style='text-align: center;'>Shape {idx}</div>"
                    html += "</td>"
            html += "</tr>"
        html += "</table>"

        wandb.log({"point_clouds_layout": wandb.Html(html)})


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



    # load params:






    criterion = torch.nn.MSELoss(reduction='mean') # hybrid_loss with chamfer_loss # reduction='none'
    


    bug_handling= "./bug_handling"
    loss_scaler= 1024.0
    global_step = 0
    for epoch in range(num_epochs):
        epoch_idx=epoch+1
        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for name, pca_theta, pca_input in tqdm_train_loader:
            # print("see name:",name)
            pca_mean_shape = the_pca_mean_shape.repeat(pca_input.shape[0], 1, 1)# # torch.Size([4, 1024, 3])
            # print("shape of pca_theta:", pca_theta.shape) # torch.Size([4, 1, 64])
            pca_theta= pca_theta.squeeze(1)


            # print("shape of pca_input:", pca_input.shape) # torch.Size([4, 1024, 3])
            mean_latents= torch.zeros_like(pca_theta)
            # batch together source and target shape for two-way loss training
            source_target_points = torch.cat([pca_mean_shape, pca_input], dim=0)
            target_source_points = torch.cat([pca_input, pca_mean_shape], dim=0)
            # print("see source_target_points shape :", source_target_points.shape)#  torch.Size([8, 1024, 3])
            source_target_latents = torch.cat([mean_latents, pca_theta], dim=0)
            target_source_latents = torch.cat([pca_theta, mean_latents], dim=0)
            # print("see target_source_latents shape :", target_source_latents.shape)#  torch.Size([8, 64])

            # print("device of source_target_points:",source_target_points.device) # cuda0

            # latent_norm= pca_theta-mean_latents
            # latent_norm_val= torch.mean(latent_norm, dim=-1)
            # print("mean of lantent change:",latent_norm_val)# gt_deform_abs


            gt_deform_abs = torch.mean(
                torch.norm(pca_mean_shape - pca_input, dim=-1)
            )

            print("gt_deform_abs change:",gt_deform_abs)




            latent_seq = torch.stack(
                [source_target_latents, target_source_latents], dim=1
            )
            print("show latent_seq:",latent_seq.shape)            
            deformed_pts, points_transformed = model(source_target_points[..., :3], latent_seq)  # Not set to via_hub.
  

            print("show deformed_pts:",deformed_pts.requires_grad)
            print("show deformed_pts shape:",deformed_pts.shape)
            print("show points_transformed shape:",points_transformed.shape)

            print("see epoch:",epoch_idx)
            print("see epoch % 100:",epoch_idx % 100)
            if epoch_idx % 50 == 0:
                points_transformed_vis= points_transformed[0]
                deformed_pts_vis= deformed_pts[0]
                deformed_pts_vis_np= deformed_pts_vis.detach().cpu().numpy()
                points_transformed_vis_np= points_transformed_vis.detach().cpu().numpy()
                deformed_pts_vis_np_pcd= o3d.geometry.PointCloud()
                points_transformed_vis_np_pcd= o3d.geometry.PointCloud()
                deformed_pts_vis_np_pcd.points=o3d.utility.Vector3dVector(deformed_pts_vis_np)
                points_transformed_vis_np_pcd.points=o3d.utility.Vector3dVector(points_transformed_vis_np)
                # o3d.visualization.draw_geometries([deformed_pts_vis_np_pcd])
                # o3d.visualization.draw_geometries([points_transformed_vis_np_pcd])
                # save the pcd:
                pcd_path= os.path.join(bug_handling,str(epoch_idx)+"_deformed_pts_vis_np_pcd.ply" )
                points_transformed_vis_np_pcd_path= os.path.join(bug_handling,str(epoch_idx)+"_points_transformed_vis_np_pcd.ply" )
                o3d.io.write_point_cloud(pcd_path,deformed_pts_vis_np_pcd )
                o3d.io.write_point_cloud(points_transformed_vis_np_pcd_path,points_transformed_vis_np_pcd )



            # cd_l1, loss_cd_l2= calc_cd(deformed_pts, target_source_points)
            # loss_cd= criterion(loss_cd_l2, torch.zeros_like(loss_cd_l2))

            # visualize the deformed points


            loss_mse = loss_scaler*criterion(deformed_pts, target_source_points)

            # loss = loss_cd+loss_mse
            loss = loss_mse
            print("see loss_mse:",loss_mse.item())



            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            scheduler.step(loss)

            
            wandb.log({"train_loss": loss.item()}, step=global_step)
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")



            # if pca_input.is_cuda:
            #     time_t2.record()
            #     torch.cuda.synchronize()  # Wait for all GPU operations to finish
            #     elapsed_time = time_t1.elapsed_time(time_t2) / 1000.0  # Convert to seconds
            # else:
            #     time_t2 = time.perf_counter()
            #     elapsed_time = time_t2 - time_t1

            # print(f"The one forward pass took: {elapsed_time:.4f} seconds")
            

        

        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches= len(dataloader_val)
            # print("num_batches:", num_batches)# 86
            for pca_recon, pca_theta, pca_input in dataloader_val:

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


        global_step += 1




        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_weights.pth')
            wandb.log({"best_val_loss": best_val_loss})

        torch.save(model.state_dict(), 'latest_model_weights.pth')

            


        




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