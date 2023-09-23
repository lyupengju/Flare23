from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import argparse
import random
import numpy as np
from monai.losses import DiceCELoss
from monai.utils import set_determinism
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from monai.data import decollate_batch 
from monai_datamodule import organ_data_module
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric,ConfusionMatrixMetric,DiceMetric
from collections import OrderedDict
import logging
import torch.optim.lr_scheduler as lr_scheduler
from models.phase1_partial3D import FasterNet
from models.phase2_smt import SMT

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Print CUDA Information
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/plyu/Projects/dataset/Dataset050_Flare2023/', help='data directory')
parser.add_argument('--exp', type=str,  default='flare_result', help='model save path')
parser.add_argument('--max_iteration', type=int,  default=80000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default= 1, help='batch_size per gpu') 
parser.add_argument('--base_lr', type=float,  default=1e-3, help='starting learning rate') 
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=50, help='random seed')
parser.add_argument('--gpu', type=list,  default=[0], help='GPU to use')
parser.add_argument('--dataset', type=str, default='flare2023', help='dataset') #'BTCV', coronary, flare2023
parser.add_argument('--phase', type=str, default='phase_1', help='dataset') #'BTCV', coronary, flare2023


args = parser.parse_args()


# Training Configuration
max_iterations = args.max_iteration
train_data_path = args.data_path
snapshot_path =  args.exp + "/"  
gpus = args.gpu
batch_size = args.batch_size
base_lr = args.base_lr 

# Logging and Deterministic Training Setup
logging.disable(logging.WARNING)
writer = SummaryWriter(snapshot_path+'/log')
if args.deterministic:  
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_determinism(seed=args.seed)

torch.cuda.empty_cache()
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

# choosing suitable architecture
if args.phase == 'phase_1':
    # model 1
    num_classes = 2
    net = FasterNet(in_chans=1,
                    num_classes=num_classes,
                    embed_dim=32,
                    decoder_dim=64,
                    dims=(32, 64, 128, 256)
                    ).cuda()
    
    # net.load_state_dict(torch.load(args.stage1_model_weight))
    print('net_stage1 load')
else:
    # model 2
    num_classes = 15
    net = SMT( img_size=96, in_chans=1, num_classes=num_classes,
    embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
    qkv_bias=True, depths=[2, 4, 6, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2).cuda()
    # net.load_state_dict(torch.load(args.stage2_model_weight))
    print('net_stage2 load')

# net = torch.nn.DataParallel(net, device_ids=[0])

# validation onehot
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
Dice_coefficient = DiceMetric(include_background=False,reduction='mean',get_not_nans=False)
HD_95 = HausdorffDistanceMetric(include_background=False,percentile=95,reduction='mean',get_not_nans=False)

# for monai
train_ds,train_loader,val_ds,val_loader = organ_data_module( task_dir  = train_data_path, dataset = args.dataset,phase = args.phase, batch_size=batch_size,cache= False,train_mode=True)
max_epoch = max_iterations//len(train_loader)+1
print("max_epoch_num: {} ".format(max_epoch))

# trainer parameter configuration
DiceCEloss = DiceCELoss(softmax=True,include_background = True, to_onehot_y = True,lambda_dice=1.0, lambda_ce=1.0) 
optimizer = optim.AdamW(net.parameters(), lr=base_lr,weight_decay=1e-5)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(max_epoch), T_mult=1,eta_min=1e-4)
scaler = torch.cuda.amp.GradScaler() 


'''Trainer'''

# Training Loop
global_step = 0
best_dice = 0
corresponding_HD = 200
best_dice_global_step = 0
epoch_num = 0
alpha = 1.0

# Training Parameters
while epoch_num <= max_epoch:
    net.train()
    step = 0
    train_loss_per_epoch=0
    
    # start training
    epoch_iterator_train = tqdm(train_loader, desc="Training", dynamic_ncols=True)
    
    # Iterate through batches for training
    for i_batch, sampled_batch in enumerate(epoch_iterator_train):
        global_step += 1
        step += 1
        
        volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

        if args.phase == 'phase_1':
            label_batch[label_batch>0] = 1
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = net(volume_batch)
            DiceCE_per_step = DiceCEloss(outputs,label_batch)

            total_loss_per_step = alpha*(DiceCE_per_step)
            writer.add_scalar('loss/DiceCE_per_step',DiceCE_per_step.item(),global_step)

        # Backpropagation and optimization
        scaler.scale(total_loss_per_step).backward()
        train_loss_per_epoch += total_loss_per_step.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator_train.set_description( "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, total_loss_per_step))
    
    # Calculate average loss per epoch
    train_loss_per_epoch /= step
    writer.add_scalar('loss/train_loss_per_epoch', train_loss_per_epoch, epoch_num)

    # Validation and Model Saving
    if  epoch_num % ( 10 if epoch_num<round(max_epoch/1.5) else 5) == 0 or global_step == max_iterations:
        net.eval()
        patch_size =  (96,96,96)
        sw_batch_size = 1
        with torch.no_grad():
            epoch_iterator_val = tqdm(val_loader, desc="validating", dynamic_ncols=True)
            for step, batch in enumerate(epoch_iterator_val):

                val_images, val_labels = batch["image"].cuda(), batch["label"].cuda()
                
                if args.phase == 'phase_1':
                    print('333')
                    label_batch[label_batch>0] = 1 
                    with torch.cuda.amp.autocast():
                        val_outputs = net(val_images)
                else:                    
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_images,patch_size,sw_batch_size,net,0.5)
                # Both predictions and labels are processed using one-hot encoding
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

                # Calculate metrics
                Dice_coefficient(val_output_convert, val_labels_convert)
                HD_95(val_output_convert, val_labels_convert)
                
            # Aggregating Mean Dice Results across the Entire Validation Data Loader
            DSC= Dice_coefficient.aggregate().item()
            # print(DSC)
            HD95= HD_95.aggregate().item()

            Dice_coefficient.reset()
            HD_95.reset()
            print(HD95)

        # Log metrics and save best model
        writer.add_scalar('metrics/DSC', DSC,global_step)
        writer.add_scalar('metrics/HD95', HD95,global_step)

        if DSC > best_dice or ((DSC == best_dice) and  (HD95 <= corresponding_HD)) :
                best_dice = DSC
                best_dice_global_step = global_step
                corresponding_HD =HD95
                save_best_mode_path = os.path.join(snapshot_path, "best_metric_model_step_"+ str(global_step)  + '.pth')
                torch.save(net.state_dict(), save_best_mode_path)
                
                print("saved new metric model")

        print("current step: {}, current mean dice: {}, mean_HD: {}, best mean dice: {:.2f} at step {} ,its HD: {:.2f}"
                        .format(global_step, DSC,HD95, best_dice, best_dice_global_step,corresponding_HD))


    # Update learning rate and increment epoch number
    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
    scheduler.step() 
    epoch_num += 1

# Training completed
print(f"train completed, best_DSC: {best_dice:.2f} at step: {best_dice_global_step} its corresponding HD  is:{corresponding_HD}")
writer.close()
torch.cuda.empty_cache()
