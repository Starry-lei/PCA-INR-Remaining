

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

import open3d as o3d

output_dir= "./output_dir"
# wandb.init(project="PCA_Remainig", entity="thesis_lei")
# PCA-Conditioned Residual SIREN MLP

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
    dataset_val = PCDataset(args, 'val')

    global_normalization= dataset_val.get_global_normalization()


    # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    # logging.info('Length of train dataset:%d', len(dataloader_train))
    logging.info('Length of validation dataset:%d', len(dataloader_val))


    
    # dataloader_train = accelerator.prepare_data_loader(dataloader_train)
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


    best_pth= "best_model_weights.pth"

    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth))
        print("loaded the best model weights")

    # exit()

    model.eval()
    best_val_loss = float('inf')   

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    creterion = torch.nn.MSELoss() # hybrid_loss with chamfer_loss # reduction='none'
    global_step = 0
    with torch.no_grad():
        for epoch in range(num_epochs):
            total_loss = 0
            tqdm_train_loader = tqdm(dataloader_val, desc=f"Epoch {epoch + 1}/{num_epochs} Testing")
            for pca_recon, pca_rep, pca_input in tqdm_train_loader:

                deformed_points = model(pca_recon, pca_rep)

                loss = 100.0*creterion(deformed_points, pca_input)

                print("val of loss:", loss) #  0.0023---> real loss:0.000023


                print("shape of deformed_points:", deformed_points.shape) # output: torch.Size([1, 1024, 3])

                pcd_1= o3d.geometry.PointCloud()
                pcd_1.points = o3d.utility.Vector3dVector(deformed_points[0].cpu().numpy())

                denormalized_deformed_points = dataset_val.denormalize_for_inference(pcd_1, global_normalization)

                # save the point cloud
                o3d.io.write_point_cloud("denormalized_deformed_points.ply", denormalized_deformed_points)

                exit()

               

                # shape of loss: torch.Size([2, 1024, 3]) 

                

                # loss= mean_flat(loss) / args.batch_size
                # print("shape of loss:", loss.shape) # shape of loss: torch.Size([2])
                # exit()

            

                total_loss += loss.item()

                
                
                tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1


        

        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches= len(dataloader_val)
            # print("num_batches:", num_batches)# 86
            for pca_recon, pca_rep, pca_input in dataloader_val:

                loss = creterion(model(pca_recon, pca_rep), pca_input)
                val_loss += loss.item()

            # val_loss /= num_batches


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