

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
from torch.optim.lr_scheduler import CosineAnnealingLR
# from chamfer3D import dist_chamfer_3D
# from fscore import fscore
import open3d as o3d


# def calc_cd(output, gt, calc_f1=False, f1_threshold=0.0001):
#     cham_loss = dist_chamfer_3D.chamfer_3DDist()
#     dist1, dist2, _, _ = cham_loss(gt, output)
#     cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
#     # cd_t = (dist1.mean(1) + dist2.mean(1))
#     cd_t = (dist1.sum(1) + dist2.sum(1))
#     if calc_f1:
#         f1, _, _ = fscore(dist1, dist2, f1_threshold)
#         return cd_p, cd_t, f1
#     else:
#         return cd_p, cd_t
    

output_dir= "./output_dir"
wandb.init(project="PCA_Remainig", entity="thesis_lei")
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


def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"Number of NaN values: {torch.isnan(tensor).sum().item()}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Min value: {tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
        print(f"Max value: {tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
        return True
    return False

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def adaptive_weight(residuals):
    # Higher weight for smaller residuals, assuming they are from simpler structures
    weights = 1 / (torch.abs(residuals) + 1e-8)  # Add small constant to avoid division by zero
    return weights / torch.max(weights)  # Normalize weights




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

    

    mean_shape= torch.tensor(dataset_train.get_mean_shape(), device= args.device, dtype=torch.float) # 1024, 3
    
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
    model = model.to(device) # PCA residual MLP

    if hasattr(model_module, 'weights_init'):
        model.apply(model_module.weights_init)

     
    best_val_loss = float('inf')   
    lr = args.lr
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))

    optimizer = getattr(optim, args.optimizer)  
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas,eps=1e-8)


    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    # optimizer,
    # max_lr=1e-4,
    # epochs=num_epochs,
    # steps_per_epoch=len(dataloader_train),
    # pct_start=0.3,  # Warm up for 30% of training
    # div_factor=25,  # Initial lr = max_lr/25
    # final_div_factor=1000  # Final lr = max_lr/1000
    # )
    
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    model, optimizer = accelerator.prepare(model, optimizer)
    
    creterion = torch.nn.MSELoss() # hybrid_loss with chamfer_loss # reduction='none', try it not mean loss reduction='none'

    loss_scale= 100.0
    global_step = 0
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")

        for pca_recon, pca_rep, pca_input, name in tqdm_train_loader:

            # pred_residual= mean_shape.unsqueeze(0).expand(pca_recon.shape[0], -1, -1)

            pred_residual = model(pca_recon, pca_rep) # dont use the deformed_points but the deformation between the estimated deformation and
      
            gt_res= pca_input-pca_recon

            loss= loss_scale* creterion(pred_residual, gt_res)
            # loss_sum_dim1 = loss_per_element.sum(dim=1) 
            # loss_per_sample = loss_sum_dim1.sum(dim=1) 
            # loss = loss_per_sample.mean()

           



            optimizer.zero_grad()
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()

            total_loss += loss.item()

            
            wandb.log({"train_loss": loss.item()}, step=global_step)
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")
            



        

        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches= len(dataloader_val)
            # print("num_batches:", num_batches)# 86
            for pca_recon, pca_rep, pca_input, name in dataloader_val:

                ref_mean_shape= mean_shape.unsqueeze(0).expand(pca_recon.shape[0], -1, -1)

                pred_residual = model(pca_recon, pca_rep)
                gt_res= pca_input-pca_recon
                loss =  loss_scale* creterion(pred_residual, gt_res)
                val_loss += loss.item()
      


            val_loss /= num_batches
            # val_loss = val_loss*1000.0
            # scheduler.step(val_loss)

        

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
    main()