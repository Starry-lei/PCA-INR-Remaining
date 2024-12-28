

from dataset.data_loader import PCDataset
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
# PCA-Conditioned Residual SIREN MLP

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
    dataset_val = PCDataset(args, 'val')

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
    model = model_module.Model(args)
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
    
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    model, optimizer = accelerator.prepare(model, optimizer)


    the_pca_mean_shape = torch.from_numpy(np.array(dataset_train.normalized_mean_shape_pcd.points)).float().to(device)
    the_pca_mean_shape = the_pca_mean_shape.unsqueeze(0)
    print("shape of the_pca_mean_shape:", the_pca_mean_shape.shape) # shape of the_pca_mean_shape: torch.Size([1,1024, 3])
    
    criterion = torch.nn.MSELoss() # hybrid_loss with chamfer_loss # reduction='none'
    global_step = 0
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for pca_recon, pca_theta, pca_input in tqdm_train_loader:

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
            # exit()
            latent_seq = torch.stack(
                [source_target_latents, target_source_latents], dim=1
            )

            
            deformed_pts = model(source_target_points[..., :3], latent_seq)  # Not set to via_hub.
            # print("see latent_seq shape :", latent_seq.shape)# torch.Size([8, 2, 64])
            # print("see deformed_pts shape :", deformed_pts.shape)

            cd_l1, loss_cd_l2= calc_cd(deformed_pts, target_source_points)
            loss_cd= criterion(loss_cd_l2, torch.zeros_like(loss_cd_l2))
            loss_mse= criterion(deformed_pts, target_source_points)
            # print("see cd_l1, cd_l2:",cd_l1, loss_cd_l2)            
            # print("see loss_cd :", loss_cd)
            # print("see loss_mse :", loss_mse)
            # print("see loss shape :", loss)
            loss = loss_cd+loss_mse


            # loss = criterion(deformed_pts, target_source_points) # geometry aware?
            # exit()
            # deformed_points = model(pca_recon, pca_rep)
            # # visualize the input and output point clouds
            # pcd_1= o3d.geometry.PointCloud()
            # pcd_1.points = o3d.utility.Vector3dVector(pca_recon[0].cpu().numpy())
            # pcd_2= o3d.geometry.PointCloud()
            # pcd_2.points = o3d.utility.Vector3dVector(pca_input[0].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd_1])
            # o3d.visualization.draw_geometries([pcd_2])
            # exit()
            # print("shape of deformed_points:", deformed_points.shape) # output: torch.Size([8, 1024, 3])
            # print("shape of pca_input:", pca_input.shape) # pca_input: torch.Size([8, 1024, 3])
            # exit()

            # loss = 100.0*creterion(deformed_points, pca_input)# shape of loss: torch.Size([2, 1024, 3]) 

            # print("val of loss:", loss) #  0.0023---> real loss:0.000023
            # loss= mean_flat(loss) / args.batch_size
            # print("shape of loss:", loss.shape) # shape of loss: torch.Size([2])
            # exit()

            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            
            wandb.log({"train_loss": loss.item()}, step=global_step)
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")
            
git checkout -b flow_deformer
            # print("shape of pca_recon:", pca_recon.shape)
            # print("shape of pca_rep:", pca_rep.shape)
            # print("shape of pca_input:", pca_input.shape)
            # shape of pca_recon: torch.Size([8, 1024, 3]) 
            # shape of pca_rep: torch.Size([8, 1, 64])
            # shape of pca_input: torch.Size([8, 1024, 3])
            # print("device of pca_recon:", pca_recon.device) # cuda:0
            # print("device of pca_rep:", pca_rep.device)
            # print("device of pca_input:", pca_input.device)

        

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

                deformed_pts = model(source_target_points[..., :3], latent_seq)  # Not set to via_hub.
                cd_l1, loss_cd_l2= calc_cd(deformed_pts, target_source_points)
                loss_cd= criterion(loss_cd_l2, torch.zeros_like(loss_cd_l2))
                loss_mse= criterion(deformed_pts, target_source_points)
                loss = loss_cd+loss_mse


                val_loss += loss.item()


            val_loss /= num_batches

        wandb.log({
            "val_loss": val_loss,
        }, step=global_step)


        global_step += 1


        # Inside your training loop, after the validation step
        # if epoch % 10 == 0:  # Log every 10 epochs
        #     # Get a batch of validation data to visualize
        #     val_batch = next(iter(dataloader_val))
        #     pca_recon_val, pca_rep_val, pca_input_val = val_batch
            
        #     with torch.no_grad():
        #         deformed_points_val = model(pca_recon_val, pca_rep_val)                
        #         # Convert to numpy and take first 8 examples
        #         deformed_points_np = deformed_points_val.cpu().numpy()[:8]  # shape: (8, 1024, 3)
        #         target_points_np = pca_input_val.cpu().numpy()[:8]         # shape: (8, 1024, 3)
                
        #         # Create a dictionary to store all point clouds
        #         point_clouds_dict = {}
                
        #         # Log all 8 shapes for both deformed and target point clouds
        #         for i in range(8):
        #             point_clouds_dict[f"deformed_shape_{i}_epoch_{epoch}"] = wandb.Object3D(deformed_points_np[i])
        #             point_clouds_dict[f"target_shape_{i}_epoch_{epoch}"] = wandb.Object3D(target_points_np[i])
                
        #         # Log all point clouds at once
        #         wandb.log(point_clouds_dict)


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