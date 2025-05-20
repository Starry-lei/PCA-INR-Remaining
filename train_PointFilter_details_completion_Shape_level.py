

from dataset.data_loader_denoising import shapelvlDataset,RandomPairSampler,paired_collate_fn
from dataset.data_loader_denoising import PCDValataset
from accelerate.utils import DistributedDataParallelKwargs
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
from sklearn.neighbors import NearestNeighbors
from utils.dist_utils import *
output_dir= "./output_dir"


def log_point_clouds_grid( point_clouds, n_rows=2, n_cols=2, step= None, accelerator=None):
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
                wandb.log({f"shape_{idx}": wandb.Object3D(point_clouds[idx])}, step=step)
                html += f"<div style='width: 250px; height: 250px;'></div>"
                html += f"<div style='text-align: center;'>Shape {idx}</div>"
                html += "</td>"
        html += "</tr>"
    html += "</table>"

    accelerator.log({"point_clouds_layout": wandb.Html(html)}, step=step)



# the input is PCA reduced 3D point cloud and the corresponding projection parameters, the output is the 3D point cloud before PCA reduction
# I would like to learn a residual prediction model to predict the residual between the PCA reduced 3D point cloud and the original 3D point cloud
# the point cloud in the dataset is already normalized by the bounding box, and all the points in the dataset are ordered, whcih means points at the same index are correspondence


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))




def normalize_and_add_curvature_based_noise_tensor(points_tensor, 
                                                   k=20, 
                                                   noise_std_min=0.005, 
                                                   noise_std_max=0.02, 
                                                   scale_factor=15000):
    """
    For each point cloud in the batch:
      1. Normalize the points by subtracting the center and dividing by the bounding sphere's radius.
      2. Add curvature-based Gaussian noise. The base noise standard deviation is chosen 
         randomly between noise_std_min and noise_std_max (which are fractions of the 
         bounding sphere's radius). Then, each point’s noise is modulated by its local curvature.
    
    Parameters:
        points_tensor (torch.Tensor): Input tensor of shape [b, 2048, 3].
        k (int): Number of nearest neighbors for curvature estimation.
        noise_std_min (float): Minimum fraction for base noise standard deviation.
        noise_std_max (float): Maximum fraction for base noise standard deviation.
        scale_factor (float): Factor to amplify the curvature values.
    
    Returns:
        torch.Tensor: Noisy point clouds of shape [b, 2048, 3] (still normalized).
    """
    b, n, d = points_tensor.shape
    assert d == 3, "Input tensor must be of shape [b, N, 3]."
    
    noisy_batch = []
    
    # Process each point cloud in the batch separately
    for i in range(b):
        # Convert the point cloud to a NumPy array
        points = points_tensor[i].cpu().numpy()  # shape: [2048, 3]
        
        # --- Normalization Step ---
        # Compute the center of the point cloud and subtract it
        center = points.mean(axis=0)
        points_centered = points - center
        
        # Compute the bounding sphere's radius (max Euclidean distance from the center)
        distances = np.linalg.norm(points_centered, axis=1)
        radius = distances.max()
        
        # Normalize the point cloud so that its bounding sphere has radius 1
        points_normalized = points_centered / radius
        
        # --- Noise Injection Step ---
        # Choose a base noise standard deviation randomly (as a fraction of the bounding sphere)
        # Since the cloud is normalized, the base noise is directly in [0.005, 0.02]
        base_noise = random.uniform(noise_std_min, noise_std_max)
        
        # For curvature-based noise, we modulate the noise per point using the local curvature.
        # Find the k nearest neighbors for each point (using the normalized points)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_normalized)
        _, indices = nbrs.kneighbors(points_normalized)
        
        # Prepare an array to store the noise sigma for each point
        noise_sigmas = np.zeros(points_normalized.shape[0])
        
        for j, neighbors in enumerate(indices):
            # Extract the local neighborhood points
            local_pts = points_normalized[neighbors]  # shape: [k, 3]
            # Compute the 3x3 covariance matrix of the local neighborhood
            cov = np.cov(local_pts.T)
            # Compute eigenvalues (which np.linalg.eigh returns in ascending order)
            eigenvalues, _ = np.linalg.eigh(cov)
            # Use the smallest eigenvalue as a proxy for local flatness (or inversely, curvature)
            curvature = eigenvalues[0]
            # print("curvature:",curvature)
            # Map curvature to noise amplitude:
            # Higher curvature leads to a larger noise standard deviation.
            noise_sigmas[j] = base_noise * (1 + curvature * scale_factor)
        
        # Generate noise for each point based on its computed sigma
        noise = np.array([np.random.randn(3) * sigma for sigma in noise_sigmas])
        # Add the noise to the normalized points
        noisy_points = points_normalized + noise
        
        noisy_batch.append(noisy_points)
    
    # Stack the processed point clouds back into a tensor of shape [b, 2048, 3]
    noisy_batch = np.stack(noisy_batch, axis=0)
    noisy_tensor = torch.tensor(noisy_batch, dtype=points_tensor.dtype, device=points_tensor.device)
    
    return noisy_tensor


def train(args, log_dir=None):
    if torch.cuda.is_available():
        print(f"Total GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA GPUs are available.")
    
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision= 'fp16', #'fp16', # no
        gradient_accumulation_steps=1,
        # log_with=args.report_to,
        project_dir=os.path.join(output_dir, "logs"),
        fsdp_plugin=None,
        # even_batches=True,
        kwargs_handlers=[init_handler],
        # ddp_kwargs={"find_unused_parameters": True},

    )
    # LOG.info(accelerator.state)

    accelerator.init_trackers(
        project_name="PCA_latent_denoiser",
        init_kwargs={"wandb": {"entity": "thesis_lei", "save_code":True}}
    )

    # wandb.init(project="PCA_latent_denoiser", entity="thesis_lei",)


    num_gpus = accelerator.state.num_processes
    print(f"Number of GPUs being used: {num_gpus}")

    num_epochs = args.epochs

    # if args.distributed:
    #     init_dist("pytorch")
    #     _, world_size = get_dist_info()
    #     args.world_size = world_size
    #     # print("show world_size:",world_size)
    #     assert args.batch_size % world_size == 0
    #     args.dataset.train.others.bs = args.total_bs // world_size






    dataset_train = shapelvlDataset(args, 'train')

    best_model_weights_path= os.path.join(log_dir, 'best_model_weights.pth')
    latest_model_weights_path=os.path.join(log_dir, 'latest_model_weights.pth')

    dataset_val= shapelvlDataset(args, 'val')

    # exit()
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
    

    # shape_mask_clusters = dataset_train.mask_clusters
    model = model_module.Model(args,)

    
    # model = model_module.Model(args, shape_mask_clusters=shape_mask_clusters, mean_shape_point_labels= dataset_train.mean_shape_point_labels)
    # try transformer model used in partSSM
    model = model.to(device) 

    if hasattr(model_module, 'weights_init'):
        model.apply(model_module.weights_init)

    best_val_loss = float('inf')   
    lr = args.lr
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))

    
    optimizer = getattr(optim, args.optimizer)  
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    # load params:
    start_epoch = 0
    load_params=  True # True # False
    if load_params:
        # checkpoint_path= "/home/stud/chengl/storage/user/PCA-INR-Remaining/exp_shapeNet_shape_lvl/dataset/denoiser_dataset/point2ssm_dgcnn_2025-02-09T16-30-30/latest_model_weights.pth" # latest_model_weights # best_model_weights
        # path= "/home/stud/chengl/storage/user/PCA-INR-Remaining/exp_shapeNet_shape_lvl/dataset/denoiser_dataset/point2ssm_dgcnn_2025-02-09T16-39-19/latest_model_weights.pth" 
        # path ="/home/stud/chengl/storage/user/PCA-INR-Remaining/exp_shapeNet_shape_lvl/dataset/denoiser_dataset/point2ssm_dgcnn_2025-02-09T20-00-35/latest_model_weights.pth"
        # checkpoint_path ="/home/stud/chengl/storage/user/PCA-INR-Remaining/exp_shapeNet_shape_lvl/dataset/denoiser_dataset/point2ssm_dgcnn_2025-02-09T21-01-46/latest_model_weights.pth"
        checkpoint_path ="/home/stud/chengl/storage/user/PCA-INR-Remaining/exp_shapeNet_shape_lvl/dataset/denoiser_dataset/point2ssm_dgcnn_2025-02-09T21-14-05/latest_model_weights.pth"
        # state_dict = torch.load(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # model.load_state_dict(state_dict, strict=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",factor=0.1, patience=10, verbose=True, min_lr=1e-5)
    accelerator.wait_for_everyone()
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # the_pca_mean_shape = torch.from_numpy(np.array(dataset_train.normalized_mean_shape_pcd.points)).float().to(device)
    # the_pca_mean_shape = the_pca_mean_shape.unsqueeze(0)
    # print("shape of the_pca_mean_shape:", the_pca_mean_shape.shape) # shape of the_pca_mean_shape: torch.Size([1,1024, 3])
    
    criterion_mse = torch.nn.MSELoss(reduction='none') # hybrid_loss with chamfer_loss # reduction='none': reduction='sum'
    



    global_step = start_epoch
    noise_std_min=0.0
    noise_std_max= 0.0 #0.005
    shape_num = 2


    # TODO:
    # adjust the number of used from 20->16, 
    # add some reg or loss to improve the denoised point cloud quality
    # we can still try chamfer loss with GT 5k points, because the network will always output ordered point cloud!

    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        epoch_idx=epoch+1
        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")

        # print(f"Epoch {epoch}, Phase: {dataset_train.phase}, Active samples: {len(dataset_train)}")
        noise_std = random.uniform(noise_std_min, noise_std_max)

        for data_tensors  in tqdm_train_loader:
            # name, gt_5k_points, pca_input
            name, gt_5k_points, pca_input = data_tensors
            pca_input= pca_input.to(device)
            gt_5k_points= gt_5k_points.to(device)
            # print("show points shape of pca_input:",pca_input.shape)
            # show points shape of pca_input: torch.Size([b, 2048, 3])


            # TODO: add noise
            # pca_input = normalize_and_add_curvature_based_noise_tensor(pca_input, k=20, noise_std_min=noise_std_min, noise_std_max=noise_std_max, scale_factor=8000)



            
            # print("show points shape of gt_5k_points:",gt_5k_points.shape)
            # show points shape of gt_5k_points: torch.Size([b, 10000, 3])
            # exit()
            pred, loss_mse = model(pca_input, gt=gt_5k_points)
            loss= loss_mse.mean()
            # print("show loss:",loss)
            # exit()
            if epoch_idx % 10 == 0:
                predis_pred= pred[:4].detach().cpu().numpy()
                print("show shape of predis_pred:",predis_pred.shape)# 4， 1024， 3
                predis_pred[:,:, [1, 2]] = predis_pred[:,:, [2, 1]]
                if accelerator.is_main_process:
                    log_point_clouds_grid(predis_pred, step=global_step, accelerator=accelerator)
               
            print("see loss_mse:",loss.item())
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()    

            accelerator.log({"train_loss": loss.item()}, step=global_step)
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")

            current_lr = optimizer.param_groups[0]['lr']
            accelerator.log({"lr": current_lr}, step=global_step)

            # accelerator.log({"lr": scheduler.get_last_lr()[0],}, step=global_step)



        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches= len(dataloader_val)
            # print("num_batches:", num_batches)# 86
            for name, gt_5k_points, pca_input in dataloader_val:
                pca_input= pca_input.to(device)

                # pca_input = normalize_and_add_curvature_based_noise_tensor(pca_input, k=20, noise_std_min=noise_std_min, noise_std_max=noise_std_max, scale_factor=10000)


                gt_5k_points= gt_5k_points.to(device)
                pred, loss_mse = model(pca_input, gt=gt_5k_points)
                loss= loss_mse.mean()
                # loss = loss_mse
                val_loss += loss.item()
            val_loss /= num_batches


        accelerator.log({"val_loss": val_loss,}, step=global_step)# , step=global_step
        

        scheduler.step(val_loss)



        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if accelerator.is_main_process:
                # torch.save(model.state_dict(), best_model_weights_path)
                checkpoint = {
                    'epoch': epoch_idx,  # Save the current epoch
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                        
                torch.save(checkpoint, best_model_weights_path)
                accelerator.log({"best_val_loss": best_val_loss}, step=global_step)
                


        global_step += 1
        if accelerator.is_main_process:
            # torch.save(model.state_dict(), latest_model_weights_path)
            checkpoint = {
                    'epoch': epoch_idx,  # Save the current epoch
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }
            torch.save(checkpoint, latest_model_weights_path)


    

            



def main():



    # print(torch.__version__)  # Check PyTorch version
    # print(torch.version.cuda)  # Check CUDA version (if available)
    # exit()

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
            os.makedirs(log_dir, exist_ok=True)

        print("log_dir:",log_dir)
        # exp_shapeNet_shape_lvl/dataset/denoiser_dataset/point2ssm_dgcnn_2025-02-09T16-28-26
        # exit()
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])
    train(args, log_dir=log_dir)


if __name__ == '__main__':

    # conda activate freereg++ strucNet/ freereg
    # python train_PointFilter.py -c cfgs/config.yaml # for leg


    # python train_PointFilter.py -c cfgs/config_part_back.yaml
    # python train_PointFilter.py -c cfgs/config_part_seat.yaml
    # python train_PointFilter_Shape_level.py -c cfgs/infer_config_eva_cosine_decay


    # python train_PointFilter_Shape_level.py -c cfgs/config_shape_lvl.yaml # for shape lvl

    # python train_PointFilter_Shape_level.py -c cfgs/config_shape_lvl_flash.yaml


    ####
    # multi-gpu
    # accelerate launch train_PointFilter_Shape_level.py -c cfgs/config_shape_lvl.yaml # for shape lvl
    # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 train_PointFilter_Shape_level.py -c cfgs/config_shape_lvl.yaml

   #  CUDA_VISIBLE_DEVICES=0,1 bash ./dist_train.sh 2 60653 -c cfgs/config_shape_lvl.yaml 

   # accelerate launch --multi_gpu --num_processes=2 train_PointFilter_Shape_level.py -c cfgs/config_shape_lvl.yaml 
   # accelerate launch --multi_gpu --num_processes=2 train_PointFilter_Shape_level.py -c cfgs/config_shape_lvl.yaml     
    main()





