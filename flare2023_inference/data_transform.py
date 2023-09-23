
# import warnings
# warnings.filterwarnings('ignore')
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
import numpy as np
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    KeepLargestConnectedComponentd,
    Spacingd,
    Resized,
    EnsureTyped,
    Invertd,
    SaveImaged,
    Activationsd,
    SpatialPadd,
RemoveSmallObjectsd,
)

image_initial_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0, b_max=255, clip=True),
            CropForegroundd(keys=["image"], source_key="image"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2),
                mode=("bilinear"),
            ),
        ])

resize_coarse = (128, 128, 128)
stage1_test_transforms =  Compose(            
            [ NormalizeIntensityd(keys=['image'], nonzero=True),
            Resized(keys=["image"], spatial_size=resize_coarse, mode=("trilinear")),
            ])

sample_patch = (96,96,96)
stage2_test_transforms = Compose(
        [
            CropForegroundd(keys=["image"], source_key="pred_ROI"),  # 取label 的 value > 0
            NormalizeIntensityd(keys=['image'], nonzero=True),

            # NormalizeIntensityd(keys=['image']),
            SpatialPadd(keys=['image'], spatial_size=sample_patch, method='end', mode='constant'),
            # 对低于spatial_size 的维度pad
        ]
    )

stage1_post_transforms = Compose([
        # EnsureTyped(keys="pred_ROI"),
        Activationsd(keys="pred_ROI", softmax=True),
        AsDiscreted(keys="pred_ROI", argmax=True),

        Invertd(
            keys="pred_ROI",  # invert the `pred` data field, also support multiple fields
            transform=stage1_test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_ROI_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        RemoveSmallObjectsd(keys="pred_ROI", min_size=20*20*20,independent_channels=True),

        # SaveImaged(keys="pred_ROI", meta_keys="pred_ROI_meta_dict", output_dir='intermediate_output',
        #            output_postfix="stage1", output_ext=".nii.gz", resample=False),

    ])

stage2_tolabel = Compose([
        # EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                       independent=True,num_components=1),  # 最大连通域 for all organs
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='final_output',
        #            output_postfix="stage2——raw", output_ext=".nii.gz", resample=False), #在invertd 已经 resample回去了
])
stage2_post_transforms =   Compose(
    [Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=stage2_test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting![](../../../AppData/Local/Temp/64b79613b0a31b0001b9a90b.png)
        ),
       
        
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='final_output',
        #            output_postfix="stage2", output_ext=".nii.gz", resample=False), #在invertd 已经 resample回去了
        ])

restore_transforms = Compose([
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=image_initial_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            # to_tensor=True,  # convert to PyTorch Tensor after inverting![](../../../AppData/Local/Temp/64b79613b0a31b0001b9a90b.png)
        ),      
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir='outputs',
                   output_postfix="", output_ext=".nii.gz", resample=False,separate_folder = False,output_dtype = np.uint8), #在invertd 已经 resample回去了
    ])
#
# '''post processing to remove tumors that are not in contact with organs'''
def remove_outlier_tumor(out):
     # out dim: H,W,D
    # print(out.shape)
    label_14_mask = out == 14
    connective_mask, num_labels = label(label_14_mask)
    # print(np.unique(connective_mask ))
    # print(np.unique(num_labels))

    # 定义结构元素（这里使用3x3x3的立方体）
    structure_element = np.ones((3, 3, 3))
    # 执行膨胀操作
    dilated_data = binary_dilation(connective_mask, structure=structure_element).astype(np.uint8)
    dilation_connective_mask, dilation_num_labels = label(dilated_data)
    # print(dilation_num_labels)
    for label_num in range(1, dilation_num_labels + 1):
        location_mask = dilation_connective_mask == label_num
        if list(np.unique(out[location_mask])) == [0,14]:
            # print(list(np.unique(out[location_mask])))
            out[location_mask] = 0
        # else:
        #     out[location_mask] = 14 # 膨胀肿瘤区域， 根据观察预测的肿瘤区域小于ground Truth    
    return out




