# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 2
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) )
# num_iters_per_epoch = int(8417 // (num_gpus * samples_per_gpu)  * 4.554)
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

# Long-Term Fusion Parameters
do_history = False
history_cat_num = 16
history_cat_conv_out_channels = 160


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

use_checkpoint = False
sync_bn = True


# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
  
depth_categories = 88

numC_Trans=32
_dim_ = 256

empty_idx = 18  # noise 0-->255
num_cls = 19  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19
img_norm_cfg = None

occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = 48
voxel_channels = [32, 32*2, 32*4]

mask2former_num_queries = num_cls
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3 # divided by ndim
mask2former_num_heads = voxel_out_channels // 16
###################
not_use_history=True
use_sequence_group_flag=not not_use_history
CE_loss_only=True
depth_stereo=True
multi_adj_frame_id_cfg = (1, 1, 1)
load_point_semantic=True
img_seg_weight=0.1

depth_loss_ce=True
depth2occ=True
sem_sup_with_prototype=True
deformable_lift=True
deform_lift_with_offset=True

wo_assign=True
length=22
use_mask_embede2=True
depth_denoise=False
depth2occ_hidden=False

use_depth_supervision=False


use_jac_loss=True
jac_context_detach=True
jax_label_sup=True
jac_weight=0.02
jac_temp=1
use_bev_feat_after_pre_process=True

jac_wo_pow2=True
wo_jac_norm=True

torch_sparse_coor=True
jac_sum=True
use_jac_non_zero_mask=True

groups=32
kernel_size=3
use_act=True

transpose_conv=True
use_single_layer=True
softmax_act=True
depth_group=4
use_channel_conv=True
channel_conv_out_norm=True

depth2occ_post_norm=True

learnable_pose=True

learnable_weight=True
learnable_weight_dim=_dim_
learnable_weight_channel_conv=True
learnable_weight_init=True

model = dict(
    type='ALOCC',
    use_depth_supervision=use_depth_supervision,
    fix_void=fix_void,
    do_history = do_history,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
    readd=True,
    ########
    not_use_history=not_use_history,
    depth_stereo=depth_stereo,
    grid_config=grid_config,
    input_size=data_config['input_size'],
    downsample=16,
    img_seg_weight=img_seg_weight,
    depth_loss_ce=depth_loss_ce,
    depth2occ=depth2occ,
    sem_sup_with_prototype=sem_sup_with_prototype,
    deform_lift_with_offset=deform_lift_with_offset,
    use_mask_embede2=use_mask_embede2,
    depth_denoise=depth_denoise,

    use_jac_loss=use_jac_loss,
    jac_context_detach=jac_context_detach,
    jax_label_sup=jax_label_sup,
    jac_weight=jac_weight,
    jac_temp=jac_temp,
    use_bev_feat_after_pre_process=use_bev_feat_after_pre_process,
    
    jac_wo_pow2=jac_wo_pow2,
    wo_jac_norm=wo_jac_norm,
    jac_sum=jac_sum,
    use_jac_non_zero_mask=use_jac_non_zero_mask,
    
    depth2occ_post_norm=depth2occ_post_norm,
    learnable_pose=learnable_pose,
    ProbAspp_learnable_weight=learnable_weight,
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
        type='CM_DepthNet', # camera-aware depth net
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        with_cp=use_checkpoint,
        loss_depth_weight=1.,
        use_dcn=False,
        stereo=depth_stereo,
        input_size=data_config['input_size'],
        bias=5.,
        mid_channels=_dim_,
        aspp_mid_channels=96,
        prototype_channels=voxel_out_channels,
        depth2occ=depth2occ,
        deform_lift_with_offset=deform_lift_with_offset,
        length=length,
        depth2occ_hidden=depth2occ_hidden,
        
        depth_group=depth_group,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        deformable_lift=deformable_lift,
        depth2occ_hidden=depth2occ_hidden,
        torch_sparse_coor=torch_sparse_coor,
        depth_group=depth_group,
        downsample=16),
    frpn=None,
    backward_projection=None,
    # pre_process=dict(
    #     type='CustomResNet3D',
    #     numC_input=numC_Trans,
    #     with_cp=use_checkpoint,
    #     num_layer=[1,],
    #     num_channels=[numC_Trans,],
    #     stride=[1,],
    #     backbone_output_ids=[0,]),
    pre_process=dict(
        type='CustomProbAspp',
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
            use_single_layer=use_single_layer,
            softmax_act=softmax_act,
            use_channel_conv=use_channel_conv,
            channel_conv_out_norm=channel_conv_out_norm,
            learnable_weight=learnable_weight,
            learnable_weight_dim=learnable_weight_dim,
            learnable_weight_channel_conv=learnable_weight_channel_conv,
            learnable_weight_init=learnable_weight_init,
            ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans *1,
        num_layer=[1, 2, 4,4],
        with_cp=use_checkpoint,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4,numC_Trans*8],
        stride=[1,2,2,2],
        backbone_output_ids=[0,1,2,3]),

    img_bev_encoder_neck=dict(type='LSSFPN3D2',
                              in_channels=numC_Trans*15,
                              out_channels=voxel_out_channels),
  
    maskformerocc_head=dict(
        type='Mask2FormerOccHead',
        feat_channels=mask2former_feat_channel,
        out_channels=voxel_out_channels,
        num_queries=mask2former_num_queries,
        num_occupancy_classes=num_cls,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        num_transformer_feat_level=0,
        # using stand-alone pixel decoder
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=mask2former_pos_channel, normalize=True),
        # using the original transformer decoder
        #########
        wo_assign=wo_assign,
        mask_embed2=use_mask_embede2,
        out_channels_embed2=numC_Trans,
        num_points_img=1056,
        ############
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=0,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        # loss settings
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * (num_cls )+ [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        point_cloud_range=point_cloud_range,
    
        train_cfg=dict(
        # pts=dict(
            num_points=12544 * 2,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler'),
        # )
        ),
    test_cfg=dict(
        pts=dict(
            semantic_on=True,
            panoptic_on=False,
            instance_on=False)),
    ),
    pts_bbox_head=None)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/gts'


train_pipeline = [
    dict(
        type='PrepareImageInputs_FBBEV',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth_FBBEV',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile_FBBEV',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        point_with_semantic=load_point_semantic,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth_FBBEV', downsample=1, grid_config=grid_config,load_semantic_map=load_point_semantic,fix_void=fix_void),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='LoadOccupancy', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs',  'gt_occupancy', 'gt_occupancy_ori', 'gt_depth','aux_cam_params','adj_aux_cam_params','gt_semantic_map'
                               ])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs_FBBEV', data_config=data_config, sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth_FBBEV',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile_FBBEV',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(type='LoadOccupancy',  occupancy_path=occupancy_path),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs',  'gt_occupancy', 'visible_mask','aux_cam_params','adj_aux_cam_params'])
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
    dict(
        type='FusionRateControlDepthHook',
        temporal_start_iter=0,
        temporal_end_iter=num_iters_per_epoch *6,
    ),
]
load_from='ckpts/bevdet-r50-4d-stereo-cbgs.pth'
