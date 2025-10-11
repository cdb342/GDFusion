# ===> per class IoU of 6019 samples:
# ===> others - IoU = 9.9809
# ===> barrier - IoU = 46.5285
# ===> bicycle - IoU = 20.3205
# ===> bus - IoU = 41.8872
# ===> car - IoU = 51.0318
# ===> construction_vehicle - IoU = 23.7781
# ===> motorcycle - IoU = 21.8902
# ===> pedestrian - IoU = 23.7551
# ===> traffic_cone - IoU = 26.1882
# ===> trailer - IoU = 31.8892
# ===> truck - IoU = 37.5055
# ===> driveable_surface - IoU = 81.0807
# ===> other_flat - IoU = 40.0958
# ===> sidewalk - IoU = 52.1753
# ===> terrain - IoU = 55.4099
# ===> manmade - IoU = 46.8645
# ===> vegetation - IoU = 40.9178
# ===> free - IoU = 90.4246
# ===> mIoU of 6019 samples: 38.31
# ===> occupied - IoU = 71.1731
# ===> mIoU_D = 31.5072
# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 2
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu))
num_epochs = 24
checkpoint_epoch_interval = 1
use_custom_eval_hook=True

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order 
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

history_cat_num = 16

# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

use_checkpoint = True
sync_bn = True


# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
depth_channels = int((grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2])

numC_Trans=32
_dim_ = 256

empty_idx = 18  # noise 0-->255
num_cls = 19  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19
voxel_out_channel = 32
occ_encoder_channels = [numC_Trans,numC_Trans*2,numC_Trans*4]
not_use_history=True
use_sequence_group_flag=not not_use_history
CE_loss_only=True
depth_stereo=True
multi_adj_frame_id_cfg = (1, 1, 1)
use_depth_supervision=False
use_camera_visible_mask=True

soft_filling=True
soft_filling_with_offset=True
use_causal_loss=True
causal_loss_weight=0.02
torch_sparse_coor=True
causal_loss_non_zero=True
groups=32
kernel_size=3
use_act=True
transpose_conv=True
softmax_act=True
geometry_group=4
use_channel_conv=True
channel_conv_out_norm=True
learnable_pose=True
normconv=True
learnable_weight_dim=_dim_
model = dict(
    type='ALOCC',
    use_depth_supervision=use_depth_supervision,
    fix_void=fix_void,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
    not_use_history=not_use_history,
    depth_stereo=depth_stereo,
    grid_config=grid_config,
    downsample=16,

    soft_filling_with_offset=soft_filling_with_offset,
    use_causal_loss=use_causal_loss,
    causal_loss_weight=causal_loss_weight,
    
    causal_loss_non_zero=causal_loss_non_zero,
    
    learnable_pose=learnable_pose,
    normconv=normconv,

    img_backbone=dict(
        # pretrained='torchvision://resnet50',
        # pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=use_checkpoint,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_channels,
        with_cp=use_checkpoint,
        loss_depth_weight=0.05,
        use_dcn=False,
        stereo=depth_stereo,
        input_size=data_config['input_size'],
        bias=5.,
        mid_channels=_dim_,
        aspp_mid_channels=96,

        soft_filling_with_offset=soft_filling_with_offset,
        geometry_group=geometry_group,
    ),
    view_transformer=dict(
        type='LSSViewTransformerFunction',
        grid_config=grid_config,
        input_size=data_config['input_size'],

        soft_filling=soft_filling,
        torch_sparse_coor=torch_sparse_coor,
        geometry_group=geometry_group,
        downsample=16),
    pre_process=dict(
        type='NormConvolution',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,],
        groups=groups,
        kernel_size=kernel_size,
        use_act=use_act,
        transpose_conv=transpose_conv,
        softmax_act=softmax_act,
        use_channel_conv=use_channel_conv,
        channel_conv_out_norm=channel_conv_out_norm,
        learnable_weight=normconv,
        learnable_weight_dim=learnable_weight_dim,
        ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans *1,
        num_layer=[1, 2, 4],
        with_cp=use_checkpoint,
        num_channels=occ_encoder_channels,
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=voxel_out_channel),
    occupancy_head= dict(
        type='OccHead_BEVDet',
        with_cp=use_checkpoint,
        use_focal_loss=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        final_occ_size=occ_size,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        
        ),
        use_deblock=False,
        CE_loss_only=CE_loss_only,
    ),
    
    pts_bbox_head=None)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/gts'

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config,fix_void=fix_void),
    dict(type='LoadOccupancy', ignore_nonvisible=use_camera_visible_mask, fix_void=fix_void, occupancy_path=occupancy_path),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs',   'gt_occupancy', 'gt_depth','aux_cam_params','adj_aux_cam_params',
                               ])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs','aux_cam_params','adj_aux_cam_params'])
            ]
        )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    occupancy_path=occupancy_path,
    use_sequence_group_flag=use_sequence_group_flag,
    stereo=depth_stereo,
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    # test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        img_info_prototype='bevdet4d',
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=use_sequence_group_flag,
        filter_empty_gt=filter_empty_gt,
        stereo=depth_stereo,
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 2e-4
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)

evaluation = dict(
    interval=num_epochs , pipeline=test_pipeline)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]


load_from='ckpts/pretrain/bevdet-r50-4d-stereo-cbgs.pth'