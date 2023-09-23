import numpy as np
from monai.utils import set_determinism
import shutil
import torch
from pathlib import Path
import os
import re
from partial3D import FasterNet
from smt import  SMT
from data_transform import stage2_tolabel,image_initial_transforms,stage1_test_transforms,stage2_test_transforms,stage1_post_transforms,stage2_post_transforms,restore_transforms,remove_outlier_tumor

import argparse
import random
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
import time
import glob
import torch.backends.cudnn as cudnn
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 是否要注视掉

'''set hyperparameters'''
parser = argparse.ArgumentParser()
parser.add_argument('--inputs_dir', type=str, default=r'D:\DATASET\flare2023\Release\flare2023\validation_image', help='dir of output')
parser.add_argument('--stage2_output_dir', type=str, default='outputs', help='dir of output')
parser.add_argument('--stage1_model_weight', type=str, default='model_stage1.pth', help='weight1')
parser.add_argument('--stage2_model_weight', type=str, default='model_stage2.pth', help='weight2')

args = parser.parse_args()
args.seed = 50
args.deterministic = True

if args.deterministic:  #设置种子.固定随机数,及相关算法,以便复现
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_determinism(seed=args.seed)


'''##################################load models##################################'''

# model 1
net_stage1 = FasterNet(in_chans=1,
                    num_classes=2,
                    embed_dim=32,
                    decoder_dim=64,
                    dims=(32, 64, 128, 256)
                    ).cuda()
net_stage1.load_state_dict(torch.load(args.stage1_model_weight))
net_stage1.eval()
print('net_stage1 load')


# model 2
net_stage2 = SMT(
embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
qkv_bias=True, depths=[2, 4, 6, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2).cuda()
# net_stage2 = torch.nn.DataParallel(net_stage2, device_ids=[0])
net_stage2.load_state_dict(torch.load(args.stage2_model_weight))
net_stage2.eval()
print('net_stage2 load')



'''##################################find all test data##################################'''
all_nii_list = sorted(glob.glob(os.path.join(args.inputs_dir, '*.nii.gz')))
''' predict one by one'''
for nii_name in all_nii_list:
    # if Path(nii_name).name.split('_0000.nii.gz')[0]+ '.nii.gz' in os.listdir(val_dir):
    # if int(re.findall("\d+",Path(nii_name).stem)[1])  in [69]:
        # print(nii_name)
        print('prediction starts!')
        '''---------------------阶段一数据加载、模型加载及推理-------------------------------------------------'''
        # start_time = time.time()

        test_image_files=[{"image": nii_name}]
        input_tensor = image_initial_transforms(test_image_files)
        input_stage1 = stage1_test_transforms(input_tensor)
        # print((input_tensor)[0]['image'].data)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for test_data in input_stage1:
                    images = test_data['image'].unsqueeze(dim=0).cuda() #add batch dim
                    test_data['pred_ROI'] = net_stage1(images).squeeze(dim=0).cpu()
                    test_data = stage1_post_transforms(test_data)
                    test_data['image'] =  input_tensor[0]['image']

                # print('stage1 prediction finished')

                # ''' determine if any forground label exist'''
                if 1  not in test_data['pred_ROI']:
    
                    print('no abdominal organs detected!!!')
                    input_tensor[0]['pred'] = test_data['pred_ROI']
                    output_tensor = restore_transforms(input_tensor)
                    del  input_tensor,output_tensor, test_data        

                else:
                    #----------------------阶段二数据加载、模型加载及推理-------------------------------------------------------------------
                        input_stage2 = stage2_test_transforms([test_data])
                        for out_data in input_stage2:
                            images = out_data['image'].unsqueeze(dim=0).cuda() #add batch dim
                            out_data['pred'] = sliding_window_inference(
                                        images, (96, 96, 96), 1, net_stage2, overlap=0.5,sw_device='cuda',device='cpu')
                            out_data['pred'] = out_data['pred'].squeeze(dim=0) # c, h,w,d
                           
                            out_data = stage2_tolabel(out_data)

                            if 14 in out_data['pred']:
                                out_data['pred'].data = remove_outlier_tumor(out_data['pred'].data.squeeze(dim=0)).unsqueeze(dim=0) #h,w,d
                            
                            out_data = stage2_post_transforms(out_data)

                            input_tensor[0]['pred'] = out_data['pred']
                            output_tensor = restore_transforms(input_tensor[0])

                        del  input_tensor,output_tensor, out_data, test_data        

        filename = Path(nii_name).name
        intermediate_path = os.path.join(args.stage2_output_dir, filename)
        new_file_name = filename.split('_0000.nii.gz')[0] + '.nii.gz'
        save_dir = os.path.join(args.stage2_output_dir, new_file_name)
        os.replace(intermediate_path, save_dir)                  
                
        # t2 = time.time()
        # print("total推理时间：{}".format(t2 - start_time))
        print('final prediction finished,next')
print('all done')
   















