# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
import mmcv
from mmcv.runner import force_fp32
import os
# from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
# from mmdet3d.models.builder import build_head, build_neck
import numpy as np
# import copy 
# import spconv.pytorch as spconv
# from tqdm import tqdm 
# from mmdet3d.models.fbbev.utils import run_time
import torch
# from torchvision.utils import make_grid
# import torchvision
# import matplotlib.pyplot as plt
# import cv2
# from collections import defaultdict
# from mmcv.runner import get_dist_info
# from mmdet.core import reduce_mean
# import mmcv
# from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
# from mmdet3d.core.bbox import box_np_ops # , corner_to_surfaces_3d, points_in_convex_polygon_3d_jit
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.models.backbones.swin_bev import SwinTransformerBEVFT
from mmdet3d.models.fbbev.modules.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.fbbev.modules.occ_loss_utils import geo_scal_loss_geo, sem_scal_loss_sem, CE_ssc_loss
from mmdet3d.models.fbbev.modules.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from mmdet3d.models.necks.cal_depth2occ import cal_depth2occ
def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class FBOCC(CenterPoint):

    def __init__(self, 
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEVFormer components
                 backward_projection=None,
                 # FB-BEV components
                 frpn=None,
                 # depth_net
                 depth_net=None,
                 # occupancy head
                 occupancy_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 readd=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,
                 #####################
                 not_use_history=False,
                 depth_stereo=False,
                 grid_config=None,
                 input_size=None,
                 downsample=16,
                 lift_attn_with_ori_feat_add=False,
                 fuse_self_round=1,
                 fuse_self=False,
                 view_trans_fuse_history=False,
                 inter_binary_non_mask=False,
                 inter_sem_geo_loss=False,
                 use_focal_loss=True,
                 loss_weight_cfg=None,
                 balance_cls_weight=True,
                 num_cls =19,
                 fuse_detach=False,
                 fuse_final_feat=False,
                 use_another_encoder=False,
                 pre_process=None,
                 maskformerocc_head=None,
                 depth2occ=False,
                 renderencoder=False,
                 maskformer_ce_loss=False,
                 maskformer_ce_loss_multi=False,
                 front_fuse=False,
                 num_channels_neck=None,
                 numC_input_neck=None,
                 use_3d_encoder=False,
                 kernel_size_front_fuse=[2,2,3],
                strid_front_fuse=[2,2,1],
                padding_front_fuse=[0,0,1],
                depth2occ_v3=False,
                encoder_deform_lift=False,
                depth2occ_v3_sup_sem_only=False,
                img_bev_encoder_neck3=None, 
                vae_occ_train_vq_vae=False,
                voxel_head=None,
                recon_loss_type='ce',
                recon_loss_weight=1.0,
                depth2occ_with_prototype=False,
                not_use_maskformerocc_head=False,
                img_seg_weight=1.0,
                sup_sem_only=False,
                decoder_2dTo3d=None,
                n_e = 512,
                dz=16,
                bev_out_channel=256,
                vq_sup=False,
                vq_mf_head=None,
                depth_loss_ce=False,
                sem_sup_with_prototype=False,
                sup_occ=False,
                direct_learn_hideen=False,
                depth2occ_composite=False,
                depth2occ_composite_min_entro_sup=False,
                semantic_cluster=False,
                semantic_prototype_decay=0.9,
                deform_lift_with_offset=False,
                use_mask_embede2=False,
                dataset='nuscenes',
                flash_occ=False,
                pred_flow=False,
                wo_pred_occ=False,
                img_backbone_only=False,
                load_freeze_feat=False,
                tta=False,
                  **kwargs):
        super(FBOCC, self).__init__(**kwargs)
        self.fix_void = fix_void
      
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None
        self.pre_process = builder.build_backbone(pre_process) if pre_process else None
        self.maskformerocc_head = builder.build_head(maskformerocc_head) if maskformerocc_head else None


        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        
        self.readd = readd # fuse voxel features and bev features
        
        self.use_depth_supervision = use_depth_supervision
        
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
       
        ################
        self.downsample=downsample
        self.grid_config=grid_config
        
        self.depth_stereo=depth_stereo
        self.num_frame=1 if not depth_stereo else 2
        self.temporal_frame=1
        self.extra_ref_frames=1 if  depth_stereo else 0
        
        
        self.not_use_history=not_use_history
        self.lift_attn_with_ori_feat_add=lift_attn_with_ori_feat_add
        self.fuse_self_round=fuse_self_round
        self.fuse_self=fuse_self
        self.fuse_detach=fuse_detach
        # self.view_trans_fuse_history=view_trans_fuse_history
        self.fuse_his_attn=view_trans_fuse_history
        self.inter_binary_non_mask=inter_binary_non_mask
        self.inter_sem_geo_loss=inter_sem_geo_loss
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))
            self.focal_loss_geo = builder.build_loss(dict(type='CustomFocalLoss',activated=True))
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        if balance_cls_weight:
            if num_cls == 19:
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls] + 0.001))
                self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
                # import pdb;pdb.set_trace()
                self.class_weights_geo = torch.cat([torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls][-1:] + 0.001)),\
                    torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls][:-1].sum() + 0.001)[None])])
                                                    
            else:
                if num_cls == 17: nusc_class_frequencies[0] += nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls] + 0.001))
        else:
            self.class_weights = torch.ones(num_cls)/num_cls  # FIXME hardcode 17
        self.fuse_final_feat=fuse_final_feat
        self.test_inter_occ=False
        self.use_another_encoder=use_another_encoder
        
        self.img_bev_encoder_backbone2 = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone and self.use_another_encoder else None
        self.img_bev_encoder_neck2 = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck and self.use_another_encoder else None
        self.depth2occ=depth2occ
        self.renderencoder=renderencoder
        self.maskformer_ce_loss=maskformer_ce_loss
        self.maskformer_ce_loss_multi=maskformer_ce_loss_multi
        self.front_fuse=front_fuse
        self.use_3d_encoder=use_3d_encoder
        if self.front_fuse:
            if self.use_3d_encoder:
                up_conv=nn.ConvTranspose3d
                conv=nn.Conv3d
            else:
                up_conv=nn.ConvTranspose2d
                conv=nn.Conv2d
            # kernel_size=[2,2,3]
            # strid=[2,2,1]
            # padding=[0,0,1]
            channels_in = numC_input_neck
            self.bev_neck=nn.ModuleList()
            self.neck_downsample=nn.ModuleList()
            for i in range(len(num_channels_neck)):
                channels_out = num_channels_neck[i]
                bev_neck=nn.Sequential(
                    up_conv(channels_in, channels_out, kernel_size_front_fuse[i],strid_front_fuse[i],  padding_front_fuse[i]),
                    nn.BatchNorm2d(channels_out),
                    nn.ReLU(inplace=True),
                    conv(channels_out, channels_out, 3, 1, 1),
                    nn.BatchNorm2d(channels_out), 
                )
                neck_downsample=up_conv(channels_in, channels_out, kernel_size_front_fuse[i],strid_front_fuse[i],  padding_front_fuse[i])
                
                channels_in = channels_out
                self.bev_neck.append(bev_neck)
                self.neck_downsample.append(neck_downsample)
                
        self.depth2occ_v3=depth2occ_v3
        self.encoder_deform_lift=encoder_deform_lift
        self.depth2occ_v3_sup_sem_only=depth2occ_v3_sup_sem_only
        self.img_bev_encoder_neck3 = builder.build_neck(img_bev_encoder_neck3) if img_bev_encoder_neck3  else None
        self.vae_occ_train_vq_vae=vae_occ_train_vq_vae
        self.voxel_head = builder.build_backbone(voxel_head) if voxel_head else None
        self.recon_loss_type=recon_loss_type
        if recon_loss_type == 'ce':
            self.recon_loss = nn.CrossEntropyLoss()
        elif recon_loss_type == 'mse':
            self.recon_loss = nn.MSELoss()
        self.num_cls=num_cls
        self.recon_loss_weight=recon_loss_weight
        self.depth2occ_with_prototype=depth2occ_with_prototype
        self.not_use_maskformerocc_head=not_use_maskformerocc_head
        self.img_seg_weight=img_seg_weight
        self.sup_sem_only=sup_sem_only
        if decoder_2dTo3d:
            if vq_mf_head is None:
                self.vq_predicter=nn.Sequential(
                    nn.Linear(bev_out_channel,bev_out_channel*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(bev_out_channel*2,n_e),
                )
        self.decoder_2dTo3d=builder.build_backbone(decoder_2dTo3d) if decoder_2dTo3d else None
        self.vq_sup=vq_sup
        self.vq_mf_head = builder.build_head(vq_mf_head) if vq_mf_head else None
        self.depth_loss_ce=depth_loss_ce
        self.sem_sup_with_prototype=sem_sup_with_prototype
        self.sup_occ=sup_occ
        self.direct_learn_hideen=direct_learn_hideen
        self.depth2occ_composite=depth2occ_composite
        self.depth2occ_composite_min_entro_sup=depth2occ_composite_min_entro_sup
        self.semantic_cluster=semantic_cluster
        if self.semantic_cluster:
            self.semantic_prototype=nn.Parameter(torch.zeros(num_cls-1,single_bev_num_channels),requires_grad=False)
            self.empty_embedding=nn.Parameter(torch.randn(1,single_bev_num_channels),requires_grad=True)
            self.semantic_prototype_decay=semantic_prototype_decay
        self.deform_lift_with_offset=deform_lift_with_offset
        self.use_mask_embede2=use_mask_embede2
        self.dataset=dataset
        self.o_var=0
        self.d_var=0
        self.o_var_max=0
        self.d_var_max=0
        self.count_var=0
        self.flash_occ=flash_occ
        self.pred_flow=pred_flow
        self.wo_pred_occ=wo_pred_occ
        self.img_backbone_only=img_backbone_only
        self.load_freeze_feat=load_freeze_feat
        self.tta=tta
        #######################################
        if not self.not_use_history and not self.fuse_final_feat:
             ## Embed each sample with its relative temporal offset with current timestep
            conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
            self.history_keyframe_time_conv = nn.Sequential(
                conv(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.SyncBatchNorm(self.single_bev_num_channels),
                nn.ReLU(inplace=True))
            ## Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.SyncBatchNorm(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def image_encoder(self, img, stereo=False,img_metas=None,x_freeze=None,**kwargs):
        imgs = img
        
        # if self.tta:
        #     imgs=imgs[[0,-1]]
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        if self.load_freeze_feat:
            # if x_freeze[1].shape[1]==1024:
                x_load=x_freeze
                load_idx_max=2
                x=self.img_backbone(imgs,load_idx_max=load_idx_max,x_load=x_load[1])
                x=(x_load[0],x_load[1],x[0])
            # else:
            #     x = self.img_backbone(imgs)
            #     save_path=os.path.join('data/nuscenes/convnex_xl_feat',img_metas[0]['scene_name'],img_metas[0]['sample_idx'])+'.pth'
            
                
            #     saved_x=dict(feat0=x[0].half(),feat2=x[1].half())
            #     torch.save(saved_x,save_path)
            #     print('saved_feat_to',save_path)
        else:

            # saved_x=dict(feat0=x[0].half(),feat2=x[1].half())
            #     torch.save(saved_x,save_path)
            x = self.img_backbone(imgs)
        # self.img_backbone.eval()

        # import pdb;pdb.set_trace()
        if self.img_backbone_only:
            return x
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        # import pdb;pdb.set_trace()
        if self.with_img_neck:
            x = self.img_neck(x)
            if self.renderencoder:
                msfeat=x
                x=x[1]
                
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        # if self.tta:
        #     x=x.unsqueeze(1).repeat(1,2,1,1,1,1).view(B*2,N, output_dim, ouput_H, output_W)
        #     # import pdb;pdb.set_trace()
        #     _,dim_s,H_s,W_s=stereo_feat.shape
        #     stereo_feat=stereo_feat.reshape(B,N, dim_s,H_s, W_s)
        #     stereo_feat=stereo_feat.unsqueeze(1).repeat(1,2,1,1,1,1).view(B*2,N, dim_s,H_s, W_s)
        #     stereo_feat=stereo_feat.view(B*2*N, dim_s, H_s, W_s)
        if self.renderencoder:
            # print(self.img_backbone.stages[2][26].depthwise_conv.weight.mean(),self.img_backbone.stages[2][26].depthwise_conv.weight.std(),self.img_backbone.stages[2][26].depthwise_conv.weight.max(),self.img_backbone.stages[2][26].depthwise_conv.weight.min())
            return x, stereo_feat,msfeat
        else:
            return x, stereo_feat
    def img_bev_encoder_backbone_front_fuse(self, x):
        # out_idx=[0,2]
        out=[]
        for i, layer in enumerate(self.bev_neck):
            if self.neck_downsample[i]:
                identity = self.neck_downsample[i](x)
            else:
                identity = x
            x = layer(x)
            x =x+ identity
            x = F.relu(x)
            # if i in out_idx:
            out.append(x)
        # import pdb;pdb.set_trace()
        # out=[torch.randn(12,704,32,88).to(x)+x.sum(),torch.randn(12,352,64,176).to(x)+x.sum(),torch.randn(12,352,64,176).to(x)+x.sum()]
        return out[::-1]

    @force_fp32()
    def bev_encoder(self, x,msfeat=None,use_another_encoder=False):
        
        # if self.front_fuse:
        #     self.img_bev_encoder_backbone= self.img_bev_encoder_backbone_front_fuse
        if self.with_specific_component('img_bev_encoder_backbone'):
            if use_another_encoder:
                x = self.img_bev_encoder_backbone2(x)
            else:
                if self.renderencoder:

                    x = self.img_bev_encoder_backbone(x,msfeat=msfeat)
                else:
                    x = self.img_bev_encoder_backbone(x)
        torch.cuda.empty_cache()
        if self.with_specific_component('img_bev_encoder_neck'):
            if use_another_encoder:
                x = self.img_bev_encoder_neck2(x)
            else:
                x = self.img_bev_encoder_neck(x)
                # import pdb;pdb.set_trace()
        
        if type(x) not in [list, tuple]:
             x = [x]
        # import pdb;pdb.set_trace()


        return x
    def bev_encoder_front_fuse(self, x,cam_params=None,depth=None,msfeat=None,use_another_encoder=False):
        
        if self.front_fuse:
            self.img_bev_encoder_backbone= self.img_bev_encoder_backbone_front_fuse
        if self.with_specific_component('img_bev_encoder_backbone'):
            if use_another_encoder:
                x = self.img_bev_encoder_backbone2(x)
            else:
                if self.renderencoder:

                    x = self.img_bev_encoder_backbone(x,msfeat=msfeat)
                else:
                    x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            if use_another_encoder:
                x = self.img_bev_encoder_neck2(x)
            else:
                x = self.img_bev_encoder_neck(x)
                # import pdb;pdb.set_trace()
        
       
        # import pdb;pdb.set_trace()
        if self.front_fuse:
            if not self.use_3d_encoder:
                bn,c,h,w=x.shape
                x=x.reshape(bn,c//44,44,h,w).contiguous()
            bevfeat=self.forward_projection.view_transform_fill_all_front_fuse( cam_params,depth,  x)
                    
        
        # bevfeat=torch.ones(2,32,200,200,16).to(x[0].device)+x[0].sum()
        if type(bevfeat) not in [list, tuple]:
             bevfeat = [bevfeat]
        # import pdb;pdb.set_trace()
        return bevfeat
    
    def bev_encoder_render(self,x,msfeat,render_config,cam_params,occ2depth,img_to_ego_coor,view_transform,use_another_encoder=False):
        if self.with_specific_component('img_bev_encoder_backbone'):
            if use_another_encoder:
                x = self.img_bev_encoder_backbone2(x)
            else:
                if self.renderencoder:

                    x = self.img_bev_encoder_backbone(x,msfeat,render_config,cam_params,occ2depth,img_to_ego_coor,view_transform)
                    x,inter_occs=x
                else:
                    x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            if use_another_encoder:
                x = self.img_bev_encoder_neck2(x)
            else:
                x = self.img_bev_encoder_neck(x)

        if self.with_specific_component('img_bev_encoder_neck3'):
            x = self.img_bev_encoder_neck3(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x,inter_occs

    @force_fp32()
    def fuse_history_multi_round(self, curr_bev, img_metas, bda,update_history=False): # align features with 3d shift
        
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        # if voxel_feat:
        #     curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        # import pdb;pdb.set_trace()
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)
        # import pdb;pdb.set_trace()
        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        # import pdb;pdb.set_trace()
        #print(seq_ids,start_of_sequence,self.history_seq_ids)
        ## Deal with first batch
        
        if not update_history:
            if self.history_bev is None:
                # self.history_bev = curr_bev.clone()
                self.history_seq_ids = seq_ids.clone()
                self.history_forward_augs = forward_augs.clone()

                # Repeat the first frame feature to be history
                
                # All 0s, representing current timestep.
                self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)

        if not update_history or self.do_history:
            self.history_bev = self.history_bev.detach()

            assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

            assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                    "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        if not update_history:
            self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            if not update_history:
                self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
                self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
                self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        if not update_history or update_history:
            ## Get grid idxs & grid2bev first.
            if voxel_feat:
                n, c_, z, h, w = curr_bev.shape

            # Generate grid
            xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
            ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
            zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
            grid = torch.stack(
                (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

            # This converts BEV indices to meters
            # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
            # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
            feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
            # if self.forward_projection:
            feat2bev[0, 0] = self.forward_projection.dx[0]
            feat2bev[1, 1] = self.forward_projection.dx[1]
            feat2bev[2, 2] = self.forward_projection.dx[2]
            feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
            feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
            feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
            feat2bev[3, 3] = 1
            feat2bev = feat2bev.view(1,4,4)

            ## Get flow for grid sampling.
            # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
            # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
            # transform to previous grid locations.
            rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                    @ torch.inverse(forward_augs) @ feat2bev)

            grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
            

            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
    
            sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        if not update_history:
            self.history_sweep_time = torch.cat(
                [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
                dim=1) # B x (1 + T)
        # if not update_history or self.do_history:
        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

    # if not update_history or update_history:
        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        if update_history:
            if self.do_history:
                self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
            self.history_sweep_time = self.history_sweep_time[:, :-1]
            self.history_forward_augs = forward_augs.clone()
            # return 
        # if voxel_feat:
        #     feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()
     
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda,update_history=True): # align features with 3d shift
        # import pdb;pdb.set_trace()
        if self.flash_occ:
            curr_bev = curr_bev.unsqueeze(-1)
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        # import pdb;pdb.set_trace()
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        # print(seq_ids,self.history_seq_ids)
        # import pdb;pdb.set_trace()
        ## Deal with first batch

        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                # import pdb;pdb.set_trace()
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        if not self.flash_occ:
            # Generate grid
            xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
            ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
            zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
            grid = torch.stack(
                (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
            # import pdb;pdb.set_trace()
            # This converts BEV indices to meters
            # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
            # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
            feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
            feat2bev[0, 0] = self.forward_projection.dx[0]
            feat2bev[1, 1] = self.forward_projection.dx[1]
            feat2bev[2, 2] = self.forward_projection.dx[2]
            feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
            feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
            feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
            # feat2bev[2, 2] = 1
            feat2bev[3, 3] = 1
            feat2bev = feat2bev.view(1,4,4)
            ## Get flow for grid sampling.
            # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
            # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
            # transform to previous grid locations.
            rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                    @ torch.inverse(forward_augs) @ feat2bev)

            grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)
        else:
             # Generate grid
            xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
            ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
            grid = torch.stack(
                (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,1,4,1)

            # This converts BEV indices to meters
            # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
            # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
            feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
            feat2bev[0, 0] = self.forward_projection.dx[0]
            feat2bev[1, 1] = self.forward_projection.dx[1]
            feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
            feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
            feat2bev[2, 2] = 1
            feat2bev[3, 3] = 1
            feat2bev = feat2bev.view(1,4,4)
            ## Get flow for grid sampling.
            # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
            # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
            # transform to previous grid locations.
            rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                    @ torch.inverse(forward_augs) @ feat2bev)

            grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

            # normalize and sample
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)

            grid = grid[:,:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 1, 2) * 2.0 - 1.0

            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        
            sampled_history_bev = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)

        
        # import pdb;pdb.set_trace()

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        if self.flash_occ:
            feats_to_return=feats_to_return.squeeze(3)
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        if self.flash_occ:
            feats_to_return=feats_to_return.unsqueeze(3)
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        if self.flash_occ:
            feats_to_return=feats_to_return.squeeze(2)
        return feats_to_return.clone()
    def prepare_inputs(self, inputs,inputs_stereo, stereo=False):
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals = inputs[1:3]
        sensor2egos_stereo, ego2globals_stereo= inputs_stereo
        # import pdb;pdb.set_trace()
        sensor2egos = sensor2egos.view(B, 1, N, 4, 4).contiguous()
        ego2globals = ego2globals.view(B, 1, N, 4, 4).contiguous()
        sensor2egos_stereo = sensor2egos_stereo.view(B, 1, N, 4, 4).contiguous()
        ego2globals_stereo = ego2globals_stereo.view(B, 1, N, 4, 4).contiguous()
        
        sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
        sensor2egos_curr = sensor2egos_cv.double()
        ego2globals_curr = ego2globals_cv.double()
        sensor2egos_adj = sensor2egos_stereo.double()
        ego2globals_adj = ego2globals_stereo.double()
        curr2adjsensor = \
            torch.inverse(ego2globals_adj @ sensor2egos_adj) \
            @ ego2globals_curr @ sensor2egos_curr
        curr2adjsensor = curr2adjsensor.float().squeeze(1)
        # curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
        # curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
        # curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
        return imgs,curr2adjsensor
    # def prepare_inputs(self, inputs, stereo=False):
    #     # split the inputs into each frame
    #     # import pdb;pdb.set_trace()
    #     B, N, C, H, W = inputs[0].shape
    #     N = N // self.num_frame
    #     imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
    #     imgs = torch.split(imgs, 1, 2)
    #     imgs = [t.squeeze(2) for t in imgs]
    #     sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
    #         inputs[1:7]
    #     # import pdb;pdb.set_trace()
    #     sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4).contiguous()
    #     ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4).contiguous()

    #     # calculate the transformation from sweep sensor to key ego
    #     keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
    #     global2keyego = torch.inverse(keyego2global.double())
    #     sensor2keyegos = \
    #         global2keyego @ ego2globals.double() @ sensor2egos.double()
    #     sensor2keyegos = sensor2keyegos.float()

    #     curr2adjsensor = [None]
    #     if stereo:
    #         sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
    #         sensor2egos_curr = \
    #             sensor2egos_cv[:, :self.temporal_frame, ...].double()
    #         ego2globals_curr = \
    #             ego2globals_cv[:, :self.temporal_frame, ...].double()
    #         sensor2egos_adj = \
    #             sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
    #         ego2globals_adj = \
    #             ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
    #         curr2adjsensor = \
    #             torch.inverse(ego2globals_adj @ sensor2egos_adj) \
    #             @ ego2globals_curr @ sensor2egos_curr
    #         curr2adjsensor = curr2adjsensor.float()
    #         curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
    #         curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
    #         curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
    #         assert len(curr2adjsensor) == self.num_frame

    #     extra = [
    #         sensor2keyegos,
    #         ego2globals,
    #         intrins.view(B, self.num_frame, N, 3, 3),
    #         post_rots.view(B, self.num_frame, N, 3, 3),
    #         post_trans.view(B, self.num_frame, N, 3)
    #     ]
    #     extra = [torch.split(t, 1, 1) for t in extra]
    #     extra = [[p.squeeze(1) for p in t] for t in extra]
    #     sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
    #     return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
    #            bda, curr2adjsensor
    def extract_stereo_ref_feat(self, x):
        # import pdb;pdb.set_trace()
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
            
        elif isinstance(self.img_backbone, SwinTransformer):
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
        elif isinstance(self.img_backbone, SwinTransformerBEVFT):
            # x = self.img_backbone.patch_embed(x)
            # hw_shape = (self.img_backbone.patch_embed.DH,
            #             self.img_backbone.patch_embed.DW)
            x, hw_shape = self.img_backbone.patch_embed(x)
            if self.img_backbone.use_abs_pos_embed:
                # x = x + self.img_backbone.absolute_pos_embed
                absolute_pos_embed = F.interpolate(self.img_backbone.absolute_pos_embed, 
                                                size=hw_shape, mode='bicubic')
                x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
        else:
            for i in range(4):
                x = self.img_backbone.downsample_layers[i](x)
                x = self.img_backbone.stages[i](x)
                # if i in self.img_backbone.out_indices:
                #     norm_layer = getattr(self.img_backbone, f'norm{i}')
                #     x_out = norm_layer(x)
                #     return x_out
                return x
    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}
        # import pdb;pdb.set_trace()
        if self.depth_stereo:
            # # import pdb;pdb.set_trace()
            # img_stereo=[img[0],*kwargs['aux_cam_params'],*img[3:]] if self.training else [img[0],*kwargs['aux_cam_params'][0],*img[3:]]
            # imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            # bda, curr2adjsensor = self.prepare_inputs(img_stereo, stereo=True)
            
            inputs_=[img[0],*kwargs['aux_cam_params']] if self.training else [img[0],*kwargs['aux_cam_params'][0]]
            inputs_stereo_=kwargs['adj_aux_cam_params'] if self.training else kwargs['adj_aux_cam_params'][0]
            # import pdb;pdb.set_trace()
            imgs, curr2adjsensor = self.prepare_inputs(inputs_,inputs_stereo_)
            # ################
            # imgs=[img[0][:,:6,...].contiguous()]
            # ##########
            # # import pdb;pdb.set_trace()
            if self.load_freeze_feat:
                save_path=os.path.join('data/nuscenes/convnex_xl_feat',img_metas[0]['scene_name'],img_metas[0]['sample_idx'])+'.pth'
                if os.path.exists(save_path):
                    # import pdb;pdb.set_trace()
                    saved_x=torch.load(save_path,map_location=imgs[0].device)
                    x_load=[saved_x['feat0'],saved_x['feat2']]
                    # import pdb;pdb.set_trace()
                    # load_idx=[0,2]
                    # load_idx_max=2
                    x_0=x_load[0]
                    x_2=x_load[1]
                    x_0=x_0.float()
                    x_2=x_2.float()
                    if x_0.shape[1]!=256 or x_0.shape[0]!=12:
                        imgs_ = img[0]
                        B, N, C, imH, imW = imgs_.shape
                        imgs_ = imgs_.view(B * N, C, imH, imW)
                        x=self.img_backbone(imgs_)
                        x_0,x_2=x[0],x[1]
                        # import pdb;pdb.set_trace()
                        flip=[kwargs['img_augs'][name][-2] for name in kwargs['img_augs'].keys()]
                        for i in range(len(flip)):
                            if flip[i]:
                                x_0=x_0.reshape(6,2,256,x_0.shape[-2],x_0.shape[-1])
                                x_0[i]=x_0[i].flip(-1)
                                x_0=x_0.reshape(12,256,x_0.shape[-2],x_0.shape[-1])
                                x_2=x_2.reshape(6,2,1024,x_2.shape[-2],x_2.shape[-1])
                                x_2[i]=x_2[i].flip(-1)
                                x_2=x_2.reshape(12,1024,x_2.shape[-2],x_2.shape[-1])

                        save_path=os.path.join('data/nuscenes/convnex_xl_feat',img_metas[0]['scene_name'],img_metas[0]['sample_idx'])+'.pth'
                        # import pdb;pdb.set_trace()
                        saved_x=dict(feat0=x[0].half(),feat2=x[1].half())
                        torch.save(saved_x,save_path)
                        print('resave',save_path,x[0].shape,x[1].shape,111111)
                        x_0,x_2=x[0],x[1]  

                    # import pdb;pdb.set_trace()
                    flip=[kwargs['img_augs'][name][-2] for name in kwargs['img_augs'].keys()]
                    for i in range(len(flip)):
                        if flip[i]:
                            x_0=x_0.reshape(6,2,256,x_0.shape[-2],x_0.shape[-1])
                            x_0[i]=x_0[i].flip(-1)
                            x_0=x_0.reshape(12,256,x_0.shape[-2],x_0.shape[-1])
                            x_2=x_2.reshape(6,2,1024,x_2.shape[-2],x_2.shape[-1])
                            x_2[i]=x_2[i].flip(-1)
                            x_2=x_2.reshape(12,1024,x_2.shape[-2],x_2.shape[-1])

                    H,W=x_0.shape[-2:]

                    x_0 =x_0.view(1, 6, 2, -1, H, W)
                    x_0 = torch.split(x_0, 1, 2)
                    x_0s = [t.squeeze(2) for t in x_0]
                    feat_prev_iv=x_0s[1]
                    stereo_feat=x_0s[0]

                    H,W=x_2.shape[-2:]
                    x_2 = x_2.view(1, 6, 2, -1, H, W)
                    x_2= x_2[:,:,0]
                    x_load=[stereo_feat.squeeze(0),x_2.squeeze(0)]
                    # print(x_load[0].shape,x_load[1].shape)

            else:
                x_load=None
            if self.renderencoder:
                context, stereo_feat,msfeat = self.image_encoder(imgs[0], stereo=True)
            else:
                context, stereo_feat = self.image_encoder(imgs[0], stereo=True,img_metas=img_metas,x_freeze=x_load,**kwargs)
                msfeat=None
            # import pdb;pdb.set_trace()
            if not  self.load_freeze_feat or feat_prev_iv.shape[1]!=256:
                with torch.no_grad():
                    # import pdb;pdb.set_trace()
                    feat_prev_iv = self.extract_stereo_ref_feat(imgs[1])
                    # import pdb;pdb.set_trace()
            stereo_metas = dict(k2s_sensor=curr2adjsensor,
                     intrins=img[3],
                     post_rots=img[4],
                     post_trans=img[5],
                    #  frustum=self.cv_frustum.to(stereo_feat.device),
                     cv_downsample=4,
                     downsample=self.downsample,
                     grid_config=self.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
            #################################################################
            # stereo_metas = None
            # import pdb;pdb.set_trace()
            # img[0]=img[0].repeat(1,2,1,1,1).contiguous()#[:,0:6,...].contiguous()
            # for i in range(5):
            #     img[i+1]=img[i+1].repeat(1,2,1,1)[:,0:6,...].contiguous() if len(img[i+1].shape)==4 else img[i+1].repeat(1,2,1).contiguous()[:,0:6,...].contiguous()
            
            # img[0]=img[0].repeat(2,1,1,1,1).contiguous()
            # for i in range(6):
            #     img[i+1]=img[i+1].repeat(2,1,1,1) if len(img[i+1].shape)==4 else img[i+1].repeat(2,1,1).contiguous()
            
            # img[0]=img[0].repeat(2,1,1,1,1).reshape(2,12,3,256,704).contiguous()
            # for i in range(5):
            #     img[i+1]=img[i+1].repeat(2,1,1,1).reshape(2,12,3,3) if len(img[i+1].shape)==4 else img[i+1].repeat(2,1,1).reshape(2,12,3).contiguous()
            # kwargs['gt_occupancy']=kwargs['gt_occupancy'].repeat(2,1,1,1)
            # kwargs['gt_depth']=kwargs['gt_depth'].repeat(2,1,1,1)
            # kwargs['aux_cam_params']=(kwargs['aux_cam_params'][0].repeat(2,1,1,1),kwargs['aux_cam_params'][1].repeat(2,1,1,1))
            
            # context,_ = self.image_encoder(img[0][:,:6,...].contiguous(), stereo=True)
            # context=context[:,:6,...].contiguous()
            # import pdb;pdb.set_trace()
            # stereo_metas = None
        else:
            if self.renderencoder:
                context, _,msfeat = self.image_encoder(img[0])
            else:
                context,_ = self.image_encoder(img[0])
                msfeat=None
            stereo_metas = None
        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            if self.depth2occ:
                context, depth,occ_weight = self.depth_net(context, mlp_input,stereo_metas,**kwargs)
           

                
                # occ_weight_save_path='occ_weight'
                # if occ_weight_save_path is not None:
                #     scene_name = img_metas[0]['scene_name']
                #     sample_token = img_metas[0]['sample_idx']
                #     # mask_camera = visible_mask[0][0]
                #     # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                #     # save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                #     save_dir=os.path.join(occ_weight_save_path, 'occ_weight',scene_name)
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     save_path = os.path.join(save_dir, f'{sample_token}.npy')
                #     np.save(save_path,occ_weight.cpu().numpy()) 

                #     kwargs['gt_depth']=kwargs['gt_depth'][0]
                #     kwargs['gt_semantic_map']=kwargs['gt_semantic_map'][0]
                #     gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                #     # gt_imgseg=gt_imgseg.reshape(occ_weight.shape[0],occ_weight.shape[1],*gt_imgseg.shape[1:])
                #     save_dir=os.path.join(occ_weight_save_path, 'semantic',scene_name)
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     save_path = os.path.join(save_dir, f'{sample_token}.npz')
                #     np.savez_compressed(save_path,gt_imgseg.cpu().numpy()) 
                
                if self.deform_lift_with_offset:
                    coor_offsets,depth=depth
                if self.depth2occ_composite:
                    length_weight,occ_weight=occ_weight
                    return_map['length_weight'] = length_weight
                elif self.depth2occ_v3:
                    predict_cls,occ_weight=occ_weight
                    return_map['image_cls'] = predict_cls
                if not self.depth2occ_v3_sup_sem_only:
                    b,n,d,h,w=depth.shape
                    # import pdb;pdb.set_trace()
                    occ=cal_depth2occ(occ_weight,depth.reshape(b*n,d,h,w))
                    occ=occ.reshape(b,n,d,h,w)
                
                # if self.sem_sup_with_prototype:
                #     if self.training:
                #         gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                #         gt_imgseg=gt_imgseg.reshape(b,n,*gt_imgseg.shape[1:])
                #     else:
                #         gt_imgseg=None
                #     mask_embede2=False if self.not_use_maskformerocc_head else True

                #     if self.semantic_cluster and self.training:
                #     # import pdb;pdb.set_trace()
                #         valid_mask=gt_imgseg!=255
                #         context_valid=context.permute(0,1,3,4,2)[valid_mask].detach()
                #         semantic_mask=gt_imgseg[valid_mask]
                #         semantic_mask=F.one_hot(semantic_mask.long(),num_classes=self.num_cls-1).float()
                #         context_cluster=torch.einsum('bn,bc->nc',semantic_mask,context_valid)/(semantic_mask.sum(dim=0)[:,None]+1)

                #         self.semantic_prototype.data = self.semantic_prototype.data * self.semantic_prototype_decay + context_cluster.data * (1 - self.semantic_prototype_decay)

                #     if self.semantic_cluster:
                #         class_prototype=torch.cat((self.semantic_prototype,self.empty_embedding),dim=0)
                #     else:
                #         class_prototype=None

                #     loss_img_seg,all_cls_scores,all_mask_preds=self.maskformerocc_head.forward_train([context.permute(0,2,1,3,4)], img_metas=img_metas, gt_labels=gt_imgseg,mask_embede2=False,class_prototype=class_prototype)
                #     return_map['loss_img_seg'] = loss_img_seg
            elif self.depth2occ_with_prototype:
                context, depth,occ_weight = self.depth_net(context, mlp_input,stereo_metas,self.maskformerocc_head,**kwargs)
                if self.deform_lift_with_offset:
                    coor_offsets,depth=depth
                b,n,d,h,w=depth.shape
                hidden_length,context_remap=occ_weight
                # import pdb;pdb.set_trace()
                if self.training:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                    gt_imgseg=gt_imgseg.reshape(b,n,*gt_imgseg.shape[1:])
                else:
                    gt_imgseg=None
                # import pdb;pdb.set_trace()
                

                loss_img_seg,all_cls_scores,all_mask_preds=self.maskformerocc_head.forward_train([context_remap], img_metas=img_metas, gt_labels=gt_imgseg)
                return_map['loss_img_seg'] = loss_img_seg
                if not self.sup_sem_only:
                    mask_pred=all_mask_preds[0].sigmoid()
                    proto_weight=mask_pred/mask_pred.sum(dim=1,keepdim=True)
                    occ_weight=torch.einsum('bcnhw,cd->bdnhw',proto_weight,hidden_length)
                    occ_weight=occ_weight.permute(0,2,1,3,4)
                    occ=cal_depth2occ(occ_weight.reshape(b*n,*occ_weight.shape[2:]),depth.reshape(b*n,d,h,w))
                    occ=occ.reshape(b,n,d,h,w)
                # import pdb;pdb.set_trace()

            else:
                context, depth = self.depth_net(context, mlp_input,stereo_metas,**kwargs)
                if self.deform_lift_with_offset:
                    coor_offsets,depth=depth
            if self.sem_sup_with_prototype and self.training:
                b,n,d,h,w=depth.shape
                if self.training:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                    gt_imgseg=gt_imgseg.reshape(b,n,*gt_imgseg.shape[1:])
                else:
                    gt_imgseg=None
                if self.use_mask_embede2:

                    mask_embede2=False if self.not_use_maskformerocc_head else True
                else:
                    mask_embede2=False

                if self.semantic_cluster and self.training:
                # import pdb;pdb.set_trace()
                    valid_mask=gt_imgseg!=255
                    context_valid=context.permute(0,1,3,4,2)[valid_mask].detach()
                    semantic_mask=gt_imgseg[valid_mask]
                    semantic_mask=F.one_hot(semantic_mask.long(),num_classes=self.num_cls-1).float()
                    context_cluster=torch.einsum('bn,bc->nc',semantic_mask,context_valid)/(semantic_mask.sum(dim=0)[:,None]+1)

                    self.semantic_prototype.data = self.semantic_prototype.data * self.semantic_prototype_decay + context_cluster.data * (1 - self.semantic_prototype_decay)

                if self.semantic_cluster:
                    class_prototype=torch.cat((self.semantic_prototype,self.empty_embedding),dim=0)
                else:
                    class_prototype=None

                loss_img_seg,all_cls_scores,all_mask_preds=self.maskformerocc_head.forward_train([context.permute(0,2,1,3,4)], img_metas=img_metas, gt_labels=gt_imgseg,mask_embede2=mask_embede2,class_prototype=class_prototype,tag='img')
                return_map['loss_img_seg'] = loss_img_seg
            # import pdb;pdb.set_trace()
            return_map['depth'] = depth
            return_map['context'] = context
            # import pdb;pdb.set_trace()
            # self.o_var=0
            # self.d_var=0
            # self.o_var_max=0
            # self.d_var_max=0
            # self.count_var=0

            # self.d_var_max+=(depth-depth.max(2,keepdim=True)[0]).pow(2).mean(2).mean().item()
            # self.o_var_max+=(occ-occ.max(2,keepdim=True)[0]).pow(2).mean(2).mean().item()
            # self.count_var+=1
            # self.o_var+=occ.var(2).mean().item()
            # self.d_var+=depth.var(2).mean().item()
            # if self.count_var%500==0:
            #     print('o_var:',self.o_var/self.count_var)
            #     print('d_var:',self.d_var/self.count_var)
            #     print('o_var_max:',self.o_var_max/self.count_var)
            #     print('d_var_max:',self.d_var_max/self.count_var)

            # import pdb;pdb.set_trace()
            # idx=np.arange(88)
            # for i in range(88): print((i,occ[0,0,:,16,16][i].item()))
            # for i in range(88): print((i,depth[0,0,:,16,16][i].item()))
            
                



                # self.semantic_prototype
            if self.sup_occ:
                if self.direct_learn_hideen:
                    return_map['occ']=depth
                else:
                    return_map['occ']=occ
            if (self.depth2occ and not self.depth2occ_v3_sup_sem_only) or (self.depth2occ_with_prototype and not self.sup_sem_only):
                depth=occ
        else:
            context=None
            depth=None
        
        if self.with_specific_component('forward_projection'):
            
            #############
            if self.lift_attn_with_ori_feat_add:
                inter_occs=[]
                # cam_params, depth, tran_feat,img_metas=None,self_bev_feat=None,fuse_self_round=0,-=None
                fuse_args = self.forward_projection.view_transform(cam_params, depth, context,img_metas=img_metas,fuse_self_round=0)
                bev_feat,depth,coor=fuse_args
                for i in range(self.fuse_self_round):
                    if self.fuse_self:
                        ###
                        if self.with_specific_component('pre_process'):
                            bev_feat = self.pre_process(bev_feat)[0]
                        if not self.not_use_history:
                            bev_feat = self.fuse_history_multi_round(bev_feat, img_metas, img[6],update_history=False)#only for round==1
                        # import pdb;pdb.set_trace()
                        
                        self_bev_feat = self.bev_encoder(bev_feat,use_another_encoder=self.use_another_encoder)
                        if self.use_another_encoder:
                            x=self_bev_feat[0]
                        else:
                        ###
                            if self.fuse_detach:
                                x=self_bev_feat[0].detach()
                            else:
                                zz=self_bev_feat[0].clone()
                                zz=zz.permute(0,2,3,4,1)
                                mask=kwargs['gt_occupancy']==255
                                zz[mask]=zz[mask].detach()
                                x=zz.permute(0,4,1,2,3)
                        # import pdb;pdb.set_trace()
                        fuse_args,inter_occ = self.forward_projection.view_transform(cam_params, depth, context,img_metas=img_metas,self_bev_feat=x,fuse_self_round=i+1,fuse_args_last_phase=fuse_args)
                    else:
                        # import pdb;pdb.set_trace()
                        fuse_args,inter_occ = self.forward_projection.view_transform(cam_params, depth, context,img_metas=img_metas,fuse_self_round=i+1,fuse_args_last_phase=fuse_args)

                        
                    inter_occs.append(inter_occ)
                    bev_feat,depth,coor=fuse_args
                return_map['inter_occs'] = inter_occs
            #################        
            
            else:
                if (self.sup_occ or self.direct_learn_hideen) and self.training:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                    kwargs['gt_imgseg']=gt_imgseg
                if not self.deform_lift_with_offset:
                    coor_offsets=None
                bev_feat = self.forward_projection(cam_params, context, depth,coor_offsets, **kwargs) 
                if (self.sup_occ or self.direct_learn_hideen) and self.training:
                    bev_feat,gt_hidden=bev_feat
                    return_map['gt_hidden']=gt_hidden
                    return_map['gt_hideen_valid_mask']=(gt_imgseg!=255).reshape(gt_hidden.shape[0],gt_hidden.shape[1],gt_hidden.shape[3],gt_hidden.shape[4])

            return_map['cam_params'] = cam_params
        else:
            bev_feat = None
        # import pdb;pdb.set_trace()
        if self.with_specific_component('frpn'): # not used in FB-OCC
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre
            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None
        # import pdb;pdb.set_trace()
        
        if self.with_specific_component('backward_projection'):

            bev_feat_refined = self.backward_projection([context],
                                        img_metas,
                                        lss_bev=bev_feat.mean(-1),
                                        cam_params=cam_params,
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth)  
                                        
            if self.readd:
                bev_feat = bev_feat_refined[..., None] + bev_feat
            else:
                bev_feat = bev_feat_refined
        del context
        torch.cuda.empty_cache()
        if self.with_specific_component('pre_process'):
            bev_feat = self.pre_process(bev_feat)[0]
            # import pdb;pdb.set_trace()
        # Fuse History
        if not self.not_use_history:
            if self.fuse_final_feat:
                bev_feat = self.forward_projection.fuse_history_final(bev_feat, img_metas, img[6],phase='two')
            else:
                if self.fuse_self:
                    bev_feat = self.fuse_history_multi_round(bev_feat, img_metas, img[6],update_history=True)
                else:
                    bev_feat = self.fuse_history(bev_feat, img_metas, img[6])
        # import pdb;pdb.set_trace()
        if self.renderencoder:
            # import pdb;pdb.set_trace()
            if self.encoder_deform_lift:
                view_transform=self.forward_projection.view_transform_core
                # import pdb;pdb.set_trace()
            else:
                view_transform=self.forward_projection.view_transform_fill_all2
            bev_feat ,inter_occs= self.bev_encoder_render(bev_feat,msfeat,render_config=self.forward_projection.render_config,cam_params=cam_params,
                                        occ2depth=self.forward_projection.occ2depth,img_to_ego_coor=self.forward_projection.img_to_ego_coor,
                                        view_transform=view_transform)
            return_map['inter_occs'] = inter_occs
        elif self.front_fuse:
            bev_feat=self.bev_encoder_front_fuse(bev_feat,cam_params=cam_params,depth=depth)

        else:
            bev_feat = self.bev_encoder(bev_feat)
        # import pdb;pdb.set_trace()
        torch.cuda.empty_cache()
        if self.with_specific_component('decoder_2dTo3d'):
            # import pdb;pdb.set_trace()
            
            vqgt=kwargs['vqgt'].unsqueeze(-1) if self.training else kwargs['vqgt'][0].unsqueeze(-1)
            if self.with_specific_component('vq_mf_head'):
                
                loss_vq,all_cls_scores,all_mask_preds=self.vq_mf_head.forward_train([bev_feat[0].unsqueeze(-1)], img_metas=img_metas, gt_labels=vqgt)
                mask_cls = F.softmax(all_cls_scores[0], dim=-1)[..., :-1]
                mask_cls=mask_cls/ mask_cls.sum(dim=-1, keepdim=True)
                mask_pred = all_mask_preds[0].sigmoid()
                prob=torch.einsum('bnhwz,bnc->bchwz',mask_pred,mask_cls)
                prob=prob/ prob.sum(dim=1, keepdim=True)
                prob=prob.squeeze(-1)
                if self.vq_sup:
                    return_map['loss_vq'] = loss_vq
       
            
            else:
                bev_feat=bev_feat[0].permute(0,2,3,1)

                logits=self.vq_predicter(bev_feat)
                if self.vq_sup:
                    loss_vq = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),vqgt.reshape(-1))
                    return_map['loss_vq'] = {'ce_loss':loss_vq}
                logits=logits.permute(0,3,1,2)
                prob=logits.softmax(dim=1)
                
            bev_feat = self.decoder_2dTo3d(prob)


        return_map['img_bev_feat'] = bev_feat
        
        # import pdb;pdb.set_trace()
        if self.fuse_his_attn:
            # import pdb;pdb.set_trace()
            if self.fuse_final_feat:
                self.forward_projection.fuse_history_final(bev_feat[0].detach(), img_metas, img[6],phase='three')
            else:
                self.forward_projection.fuse_history(bev_feat[0].detach(), img_metas, cam_params[5],update_history=True)
        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat

    def extract_vox_bev_feat(self,occ,img_metas, **kwargs):
        return self.voxel_head(occ, img_metas, **kwargs)

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        if self.img_backbone_only:
            # import pdb;pdb.set_trace()
            
            def is_file_larger_than_500mb(file_path):
                # 检查文件是否存在
                if not os.path.isfile(file_path):
                    print("File does not exist.")
                    return False

                # 获取文件大小，单位是字节
                file_size_bytes = os.path.getsize(file_path)

                # 将文件大小转换为MB
                file_size_mb = file_size_bytes / (1024 * 1024)

                # 比较文件大小
                if file_size_mb > 500:
                    return True
                else:
                    return False
            save_path=os.path.join('data/nuscenes/convnex_xl_feat',img_metas[0]['scene_name'],img_metas[0]['sample_idx'])+'.pth'

            saved_x=torch.load(save_path,map_location=img[0].device)
            x_load=[saved_x['feat0'],saved_x['feat2']]
            # import pdb;pdb.set_trace()
            # load_idx=[0,2]
            # load_idx_max=2
            x_0=x_load[0]
            x_2=x_load[1]
            # x_0=x_0.float()
            # x_2=x_2.float()
            if x_0.shape[1]!=256 or x_0.shape[0]!=12:
                        
            # if  not is_file_larger_than_500mb(save_path):
                
                x=self.image_encoder(img[0], stereo=True,img_metas=img_metas,**kwargs)
                # import pdb;pdb.set_trace()


                # os.makedirs(os.path.dirname(save_path),exist_ok=True)
                # import pdb;pdb.set_trace()
                saved_x=dict(feat0=x[0].half(),feat2=x[1].half())
                torch.save(saved_x,save_path)
                # print(os.path.getsize(save_path)/1024/1024)
                
                return dict(img_bev_feat=x)
            else:
                return dict(img_bev_feat=None)
                    ###############



        ###############

        results={}
        if img is not None and self.with_specific_component('img_backbone'):
            # self.occupancy_save_path='occupancy_vis/swin_base_80088'
            # if self.occupancy_save_path is not None:
            #     scene_name = img_metas[0]['scene_name']
            #     sample_token = img_metas[0]['sample_idx']
            #     # mask_camera = visible_mask[0][0]
            #     # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
            #     # save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
            #     save_dir=os.path.join(self.occupancy_save_path, 'occupancy_pred',scene_name)
            #     # if not os.path.exists(save_dir):
            #     #     # os.makedirs(save_dir,exists_ok=True)
            #     #     mmcv.mkdir_or_exist(save_dir)
            #     save_path = os.path.join(save_dir, f'{sample_token}.npz')
            #     if  os.path.exists(save_path) and os.path.getsize(save_path)>10:
            #         print('save',save_path)
            #         return dict(img_bev_feat=None)
            #     else:
            #         pass
            #         # os.makedirs(save_dir,exist_ok=True)
            #         # np.savez_compressed(save_path,img[0].cpu().numpy())
            # else:
            #     pass
                    
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)
        if hasattr(kwargs,'gt_occupancy') and self.with_specific_component('voxel_head'):
            results.update(self.extract_vox_bev_feat(kwargs['gt_occupancy_ori'],img_metas, **kwargs))

        return results


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # kwargs['gt_occupancy']=kwargs['gt_occupancy'].repeat(2,1,1,1)
        # kwargs['gt_depth']=kwargs['gt_depth'].repeat(2,1,1,1)
        # import pdb;pdb.set_trace()
        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if  self.with_pts_bbox:
            losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
            losses.update(losses_pts)

        if self.with_specific_component('maskformerocc_head') and not self.not_use_maskformerocc_head:
            # gt_occupancy_ori=kwargs['gt_occupancy_ori'] if 'gt_occupancy_ori' in kwargs else None
            if self.maskformer_ce_loss:
                
                gt_occupancy_ori=kwargs['gt_occupancy_ori'] if 'gt_occupancy_ori' in kwargs else None                                                     
                if self.maskformer_ce_loss_multi:
                    predict_voxel=self.maskformerocc_head.train_with_multi_layer_ce_loss(results['img_bev_feat'], img_metas=img_metas, gt_labels=kwargs['gt_occupancy'], 
                                                                     **kwargs)
                    predict_voxel=predict_voxel['output_voxels']
                    # import pdb;pdb.set_trace()
                    losses_occupancy = self.maskformerocc_head.loss_ce(target_voxels=kwargs['gt_occupancy'],
                            gt_occupancy_ori=gt_occupancy_ori,
                    output_voxels = predict_voxel,
                    )
                else:
                    predict_voxel=self.maskformerocc_head.simple_test(results['img_bev_feat'], img_metas=img_metas, gt_labels=kwargs['gt_occupancy'], 
                                                                     **kwargs)
                    predict_voxel=(predict_voxel['output_voxels'][0]+1e-8).log()
                    
                    losses_occupancy = self.maskformerocc_head.loss_ce(target_voxels=kwargs['gt_occupancy'],
                            gt_occupancy_ori=gt_occupancy_ori,
                    output_voxels = [predict_voxel],
                    )
            else:
                if self.semantic_cluster:
                    class_prototype=torch.cat((self.semantic_prototype,self.empty_embedding),dim=0)
                else:
                    class_prototype=None
                # import pdb;pdb.set_trace()
                losses_occupancy,_,_ = self.maskformerocc_head.forward_train(results['img_bev_feat'], img_metas=img_metas, gt_labels=kwargs['gt_occupancy'], 
                                                                     class_prototype=class_prototype,bda=img_inputs[6],**kwargs)

            losses.update(losses_occupancy)

        if self.with_specific_component('occupancy_head'):
            gt_occupancy_ori=kwargs['gt_occupancy_ori'] if 'gt_occupancy_ori' in kwargs else None
            losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results,**kwargs)
            losses.update(losses_occupancy)
        if self.renderencoder:
            for i in range(len(results['inter_occs'])):
                # import pdb;pdb.set_trace()
                # loss_occ = self.loss_single(voxel_semantics, mask_camera, inter_occs[i])
                # if self.inter_sup_non_mask:
                #     mask_camera=None
                
                # import pdb;pdb.set_trace()
                target_voxels=kwargs['gt_occupancy_ori']
                B, C, H, W, D = results['inter_occs'][i].shape
                ratio = target_voxels.shape[2] // H
                if ratio != 1:
                    # import pdb;pdb.set_trace()
                    target_voxels=target_voxels[:,::ratio,::ratio,::ratio]

                    mask_camera=kwargs['gt_occupancy']!=255
                    # mask = mask.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3).sum(-1) > 0
                    mask_camera=mask_camera[:,::ratio,::ratio,::ratio]
                else:
                    mask_camera=kwargs['gt_occupancy']!=255
                
                loss_occ =self.TwoPart_loss(results['inter_occs'][i].permute(0,2,3,4,1),target_voxels,mask=mask_camera)

                loss_occ = {key+str(i):value for key,value in loss_occ.items()}
                losses.update(loss_occ)
            #######################
        # for i in range(len(img_metas)):
        #     # if os.path.join(img_metas[i]['sample_idx']+'.npy') in os.listdir('data/nuscenes/gts_test_zz'):
        #         # occ_gt_ = np.load(os.path.join('data/nuscenes/gts_test_zz',img_metas[i]['sample_idx']+'.npy'))
        #     occ_gt2 = np.load(os.path.join('data/nuscenes/gts',img_metas[i]['scene_name'],img_metas[i]['sample_idx'],'labels.npz'))['semantics'][None]
        #     # print((torch.Tensor(occ_gt_).to(gt_occ)-gt_occ[i:i+1]).sum(),img_metas[i]['sample_idx'],111)
        #     print((torch.Tensor(occ_gt2).to(kwargs['gt_occupancy'])-kwargs['gt_occupancy'][i:i+1]).sum(),img_metas[i]['sample_idx'],222)
            ##################
        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)

        if self.use_depth_supervision and self.with_specific_component('depth_net') and not self.direct_learn_hideen:
            if self.depth_loss_ce:
                loss_depth = self.depth_net.get_depth_loss_(kwargs['gt_depth'], results['depth'])
            else:
                loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)
        if self.lift_attn_with_ori_feat_add:
            for i in range(len(results['inter_occs'])):
                # import pdb;pdb.set_trace()
                # loss_occ = self.loss_single(voxel_semantics, mask_camera, inter_occs[i])
                # if self.inter_sup_non_mask:
                #     mask_camera=None
                if self.inter_sem_geo_loss:
                    loss_occ =self.sem_geo_loss(results['inter_occs'][i],kwargs['gt_occupancy'],kwargs['gt_occupancy_ori'],mask=kwargs['gt_occupancy']!=255)
                else:
                    loss_occ =self.TwoPart_loss(results['inter_occs'][i].permute(0,2,3,4,1),kwargs['gt_occupancy_ori'],mask=kwargs['gt_occupancy']!=255)

                loss_occ = {key+str(i):value for key,value in loss_occ.items()}
                losses.update(loss_occ)
        if self.depth2occ_v3:
            # gt_imgseg=kwargs['gt_semantic_map']
            gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
            pred_imgseg=results['image_cls']
            valid_mask=torch.max(gt_depth, dim=1).values > 0.0
            # import pdb;pdb.set_trace()
            gt_imgseg=gt_imgseg.reshape(-1)
            pred_imgseg=pred_imgseg.permute(0,2,3,1).reshape(-1,pred_imgseg.shape[1])
            gt_imgseg=gt_imgseg[valid_mask]
            pred_imgseg=pred_imgseg[valid_mask]
            # import pdb;pdb.set_trace()
            loss_img_sem=F.cross_entropy(pred_imgseg,gt_imgseg.long())
            losses['loss_img_sem']=loss_img_sem

            # import pdb;pdb.set_trace()
        if self.with_specific_component('voxel_head'):
            losses['embed_loss']=results['embed_loss']
            if  not self.with_specific_component('maskformerocc_head'):
                target_voxels=kwargs['gt_occupancy']
                recon=results['recon']
                ##########################
                preds=recon.permute(0, 2, 3, 4, 1)
                mask=target_voxels!=255
                preds=preds[mask]
                target_voxels=target_voxels[mask]
                if self.recon_loss_type=='ce':
                    losses['loss_voxel_ce'] = self.recon_loss(preds, target_voxels)*self.recon_loss_weight
                elif self.recon_loss_type=='mse':

                    target_voxels=F.one_hot(target_voxels.long(),num_classes=self.num_cls).float()
                    preds=F.softmax(preds,dim=-1)
                    losses['loss_voxel_mse'] = self.recon_loss(preds, target_voxels)*self.recon_loss_weight
        

            ##################
        if self.depth2occ_with_prototype or self.sem_sup_with_prototype:
            for loss in results['loss_img_seg']:
                losses.update({'img_seg_'+loss:results['loss_img_seg'][loss]* self.img_seg_weight})
            
        if self.vq_sup:
            for loss in results['loss_vq']:
                losses.update({'loss_vq_'+loss:results['loss_vq'][loss]})

        if self.sup_occ:
            # import pdb;pdb.set_trace()
            occ=results['occ'].permute(0,1,3,4,2)
            gt_hidden=results['gt_hidden'].permute(0,1,3,4,2)
            valid_mask=results['gt_hideen_valid_mask']


            loss_hidden=F.binary_cross_entropy(occ[valid_mask],gt_hidden[valid_mask])
            losses['loss_hidden']=loss_hidden
        if self.depth2occ_composite_min_entro_sup:
            length_weight=results['length_weight']
            loss_min_entro=-(length_weight*(length_weight+1e-8).log()).sum(1).mean()
            losses['hidden_loss_min_entro']=loss_min_entro
        del results,kwargs,points,img_metas,gt_bboxes_3d,gt_labels_3d,gt_labels,gt_bboxes,img_inputs,proposals,gt_bboxes_ignore,gt_occupancy_flow
        torch.cuda.empty_cache()
        return losses
    
    @force_fp32() 
    def sem_geo_loss(self,inputs,target,target_ori,mask=None,occ_depth=None,occ_depth_gt=None):
        loss_dict = dict()
        target_sem=target.clone()
        target_sem[target_sem==18]=255
        inputs[:,-1]=inputs[:,-1].sigmoid()
        # if self.inter_sem_geo_loss:
        targets_occ=(target_ori!=18).float()
        # else:
        #     targets_occ=(target!=18).float()
        if self.use_focal_loss:
            # import pdb;pdb.set_trace()
            loss_dict['loss_voxel_ce_sem_in'] = self.loss_voxel_ce_weight * self.focal_loss(inputs[:,:-1], target_sem, self.class_weights[:-1].type_as(inputs), ignore_index=255)
            loss_dict['loss_voxel_ce_occ_in'] = self.loss_voxel_ce_weight * self.focal_loss_geo(torch.cat((1-inputs[:,-1:],inputs[:,-1:]),dim=1), targets_occ, self.class_weights_geo.type_as(inputs), ignore_index=255)
        else:
            loss_dict['loss_voxel_ce_in'] = self.loss_voxel_ce_weight * CE_ssc_loss(inputs[:,:-1], target_sem, self.class_weights[:-1].type_as(inputs), ignore_index=255)


        inputs[:,:-1]=inputs[:,:-1].softmax(1)
        
        loss_dict['loss_voxel_sem_scal_in'] = self.loss_voxel_sem_scal_weight * sem_scal_loss_sem(inputs, target, mask=mask)
        loss_dict['loss_voxel_geo_scal_in'] = self.loss_voxel_geo_scal_weight * geo_scal_loss_geo(inputs[:,-1], targets_occ, mask=mask)
        
        loss_dict['loss_voxel_lovasz_in'] = self.loss_voxel_lovasz_weight * lovasz_softmax(inputs[:,:-1], target_sem, ignore=255)
        loss_dict['loss_voxel_lovasz_occ_in'] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.cat((1-inputs[:,-1:],inputs[:,-1:]),dim=1), targets_occ, ignore=None)
        return loss_dict
    
    @force_fp32() 
    def TwoPart_loss(self,inputs,targets,mask=None,occ_depth=None,occ_depth_gt=None):
        # import pdb;pdb.set_trace()
        loss = dict()
        # if self.add_occ_depth_loss:
        #     # import pdb;pdb.set_trace()
        #     depth_loss_occ=self.img_view_transformer.get_depth_loss_gt_occ2depth( occ_depth+1e-8,occ_depth_gt.permute(0,2,3,1).reshape(-1,occ_depth_gt.shape[1]))
        #     # depth_loss_occ=-((occ_depth+1e-8).log()*occ_depth_gt).sum(1).mean()
        #     loss['depth_loss_occ']=depth_loss_occ
        if mask is not None:
            mask=mask.bool()
            inputs_=inputs[mask]
            targets_=targets[mask]
        else:
            inputs_=inputs.reshape(-1,inputs.shape[-1])
            targets_=targets.reshape(-1)
        sem=inputs_[:,:-1]
        occ=inputs_[:,-1]

        targets_occ=targets_!=18
        
        
        # mask_free=(targets==18).nonzero().squeeze()

        
        # if not self.only_sup_sem:
        if self.inter_binary_non_mask:
            targets=targets.reshape(-1)
            targets=targets!=18
            occ=inputs.reshape(-1,inputs.shape[-1])[:,-1]
            loss_occ=F.binary_cross_entropy_with_logits(occ,targets.float())
        else:
            loss_occ=F.binary_cross_entropy_with_logits(occ,targets_occ.float())
            # sem=(sem+1e-6).log()
            
        loss['loss_in_occ'] = loss_occ*0.1
        # if not self.sup_binary:
        
        
        mask_occ=(targets_occ).nonzero().squeeze()
        sem=sem[mask_occ]
        loss_sem=F.cross_entropy(sem,targets_[mask_occ])
        loss['loss_in_sem'] = loss_sem*0.1
        # loss=loss_occ+loss_sem
        # print((sem.sum(1)==0).sum(),2222222222222222222,loss_occ,loss_sem)
        return loss
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if self.dataset=='kitti':
            img_inputs=[img_inputs]
            img_metas=[img_metas]
            points=[points]
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            # import pdb;pdb.set_trace()
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))
            # import pdb;pdb.set_trace()
            
            if isinstance(img_metas[0],mmcv.parallel.DataContainer):
                if num_augs==1 and not img_metas[0].data[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                    return self.simple_test(points[0], img_metas[0].data[0][0], img_inputs[0],
                                        **kwargs)
                else:
                    return self.aug_test(points, img_metas, img_inputs, **kwargs)
            else:
                if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                    return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                        **kwargs)
                else:
                    return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
      



    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # import pdb;pdb.set_trace()
        
        # print(kwargs['gt_occupancy'].shape,kwargs['gt_occupancy'].mean(),kwargs['gt_occupancy'].std(),555555555555,_rank)
        # import pdb;pdb.set_trace()
        if 'semantic_anything_map' in kwargs:
            # import pdb;pdb.set_trace()
            kwargs['semantic_anything_map']=[torch.cat(kwargs['semantic_anything_map'],dim=0)]
        if 'depth_anything_map' in kwargs:
            kwargs['depth_anything_map']=[torch.cat(kwargs['depth_anything_map'],dim=0)]
        img_inputs=[torch.cat([img_inputs[i][j] for i in range(len(img_inputs))],dim=0) for j in range(len(img_inputs[0]))]
        
        img=img_inputs
        kwargs['aux_cam_params']=[[torch.cat([kwargs['aux_cam_params'][i][j] for i in range(len(kwargs['aux_cam_params']))],dim=0) for j in range(len(kwargs['aux_cam_params'][0]))]]
        # import pdb;pdb.set_trace()
        kwargs['adj_aux_cam_params']=[[torch.cat([kwargs['adj_aux_cam_params'][i][j] for i in range(len(kwargs['adj_aux_cam_params']))],dim=0) for j in range(len(kwargs['adj_aux_cam_params'][0]))]]
        # import pdb;pdb.set_trace()
        img_metas=[img_metas[i][0] for i in range(len(img_metas))]
        results = self.extract_feat(points, img=img_inputs, img_metas=img_metas, **kwargs)
        del img_inputs,points
        torch.cuda.empty_cache()
        
        pred_occupancy_category=None
        pred_density=None
        pred_flow=None
        bbox_list = [dict() for _ in range(len(img_metas))]
        if  self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]
        
        if self.with_specific_component('maskformerocc_head'):
            if self.semantic_cluster:
                class_prototype=torch.cat((self.semantic_prototype,self.empty_embedding),dim=0)
            else:
                class_prototype=None
            # import pdb;pdb.set_trace()
            output_test = self.maskformerocc_head.simple_test(results['img_bev_feat'], img_metas=img_metas,class_prototype=class_prototype, bda=img[6],**kwargs)
            torch.cuda.empty_cache()
            pred_occupancy = output_test['output_voxels'][0]
            for i in range(len(img_metas)):
                tta_config = kwargs['tta_config'][i]
                flip_dx=tta_config['flip_dx']
                flip_dy=tta_config['flip_dy']
                pred_occupancy_i = pred_occupancy[i]
                if flip_dx:
                    pred_occupancy_i = torch.flip(pred_occupancy_i, [2])
    
                if flip_dy:
                    pred_occupancy_i = torch.flip(pred_occupancy_i, [1])
                pred_occupancy[i] = pred_occupancy_i
            pred_occupancy = pred_occupancy.mean(0,keepdim=True)
            # import pdb;pdb.set_trace()
            if self.pred_flow:
                # import pdb;pdb.set_trace()
                pred_flow = output_test['output_flow'][0]
                for i in range(len(img_metas)):
                    tta_config = kwargs['tta_config'][i]
                    flip_dx=tta_config['flip_dx']
                    flip_dy=tta_config['flip_dy']
                    pred_flow_i = pred_flow[i]
                    # import pdb;pdb.set_trace()
                    if flip_dx:
                        pred_flow_i = torch.flip(pred_flow_i, [2])
        
                    if flip_dy:
                        pred_flow_i = torch.flip(pred_flow_i, [1])
                    pred_flow[i] = pred_flow_i
                pred_flow = pred_flow.mean(0,keepdim=True)




                
                pred_flow = pred_flow[0]
                # pred_flow = pred_flow.permute(0, 2, 3, 4, 1)[0]
                pred_flow=pred_flow.permute(3, 2, 0, 1)
                pred_flow = torch.flip(pred_flow, [2])
                pred_flow=pred_flow[[1,0]]
                pred_flow = torch.rot90(pred_flow, -1, [2, 3])
                pred_flow = pred_flow.permute(2, 3, 1, 0)


            
            if not self.wo_pred_occ:
                #################
                pred_density=output_test['output_density'][0]
                for i in range(len(img_metas)):
                    tta_config = kwargs['tta_config'][i]
                    flip_dx=tta_config['flip_dx']
                    flip_dy=tta_config['flip_dy']
                    pred_density_i = pred_density[i]
                    if flip_dx:
                        pred_density_i = torch.flip(pred_density_i, [1])
                    
                    if flip_dy:
                        # import pdb;pdb.set_trace()
                        pred_density_i = torch.flip(pred_density_i, [0])
                    pred_density[i] = pred_density_i
                pred_density = pred_density.mean(0,keepdim=True)
                
                pred_density=pred_density.unsqueeze(1)
                pred_density=pred_density.permute(0, 2, 3, 4, 1)[0]
                pred_density = pred_density.permute(3, 2, 0, 1)
                pred_density = torch.flip(pred_density, [2])
                pred_density = torch.rot90(pred_density, -1, [2, 3])
                pred_density = pred_density.permute(2, 3, 1, 0).squeeze(-1)

                ############
                if self.dataset=='kitti':
                    # import pdb;pdb.set_trace()
                    output={}
                    output['output_voxels'] = pred_occupancy
                    output['target_voxels'] = kwargs['gt_occupancy']
                    return output
                pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                
                if self.fix_void:
                    pred_occupancy = pred_occupancy[..., 1:]     
                    # import pdb;pdb.set_trace()

                # convert to CVPR2023 Format
                pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                pred_occupancy = torch.flip(pred_occupancy, [2])
                pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
                
                if 'vis_class' in kwargs:
                    ##########
                    # import pdb;pdb.set_trace()
                    mask=kwargs['gt_occupancy'][0]==kwargs['vis_class']
                    # mask=mask*(kwargs['gt_occupancy_ori'][0]!=255)
                    pred_occupancy=pred_occupancy[mask[0]].sum(0)
                    
                    # pred_occupancy=pred_occupancy.mean((0,1,2))

                    return pred_occupancy
                if return_raw_occ:
                    pred_occupancy_category = pred_occupancy
                    return pred_occupancy_category
                else:
                    pred_occupancy_category = pred_occupancy.argmax(-1) 
                    # pred_occupancy_category = pred_occupancy[...,:-1].argmax(-1) 
                    # pred_density=(pred_occupancy.argmax(-1)!=16).float()
                pred_occupancy_category = pred_occupancy_category.cpu().numpy().astype(np.uint8)

                # self.occupancy_save_path='occupancy_vis/swinbase_80088_pred2render_flow'
                if self.occupancy_save_path is not None:
                    scene_name = img_metas[0]['scene_name']
                    sample_token = img_metas[0]['sample_idx']
                    # mask_camera = visible_mask[0][0]
                    # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    # save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                    save_dir=os.path.join(self.occupancy_save_path, 'occupancy_pred',scene_name)
                    if not os.path.exists(save_dir):
                        # os.makedirs(save_dir,exists_ok=True)
                        mmcv.mkdir_or_exist(save_dir)
                    save_path = os.path.join(save_dir, f'{sample_token}.npz')
                    ##################
                    # # convert to CVPR2023 Format
                    # # import pdb;pdb.set_trace()
                    # occupancy_original = kwargs['gt_occupancy'][0][0].permute(2, 0, 1)
                    # occupancy_original = torch.flip(occupancy_original, [1])
                    # occupancy_original = torch.rot90(occupancy_original, -1, [1, 2])
                    
                    # occupancy_original = occupancy_original.permute(1, 2, 0)

                    # occ_flow = kwargs['gt_occ_flow'][0][0].permute(2, 0, 1,3)
                    # occ_flow = torch.flip(occ_flow, [1])
                    # occ_flow = torch.rot90(occ_flow, -1, [1, 2])
                    
                    # occ_flow = occ_flow.permute(1, 2, 0,3)

                    # np.savez_compressed(save_path,pred_occ= pred_occupancy_category,pred_flow=pred_flow.cpu().numpy(),gt_occ=occupancy_original.cpu().numpy(),gt_flow=occ_flow.cpu().numpy())
                   
                    np.savez_compressed(save_path,pred_occupancy_category)
                    # save_path_flow = os.path.join(save_dir, f'{sample_token}_flow.npz')
                    # np.savez_compressed(save_path_flow,pred_flow.cpu().numpy())
                # import pdb;pdb.set_trace()
           
            else:
                pred_occupancy_category =  None
                  #####################
                # self.occupancy_save_path='occupancy_vis/swinbase_80088_pred2render_flow'
                # if self.occupancy_save_path is not None:
                #     scene_name = img_metas[0]['scene_name']
                #     sample_token = img_metas[0]['sample_idx']
                #     # mask_camera = visible_mask[0][0]
                #     # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                #     # save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                #     save_dir=os.path.join(self.occupancy_save_path, 'occupancy_pred',scene_name)
                #     if not os.path.exists(save_dir):
                #         # os.makedirs(save_dir,exists_ok=True)
                #         mmcv.mkdir_or_exist(save_dir)
                #     save_path = os.path.join(save_dir, f'{sample_token}.npz')
                #     occupancy_original = kwargs['gt_occupancy'][0][0].permute(2, 0, 1)
                #     occupancy_original = torch.flip(occupancy_original, [1])
                #     occupancy_original = torch.rot90(occupancy_original, -1, [1, 2])
                    
                #     occupancy_original = occupancy_original.permute(1, 2, 0)

                #     occ_flow = kwargs['gt_occ_flow'][0][0].permute(2, 0, 1,3)
                #     occ_flow = torch.flip(occ_flow, [1])
                #     occ_flow = torch.rot90(occ_flow, -1, [1, 2])
                    
                #     occ_flow = occ_flow.permute(1, 2, 0,3)

                #     np.savez_compressed(save_path,pred_flow=pred_flow.half().cpu().numpy())
                #     # import pdb;pdb.set_trace()

                #     # np.savez_compressed(save_path,pred_flow=pred_flow.cpu().numpy(),gt_occ=occupancy_original.cpu().numpy(),gt_flow=occ_flow.cpu().numpy())

                #     # np.savez_compressed(save_path,pred_flow=pred_flow.cpu().numpy(),gt_occ=occupancy_original.cpu().numpy(),gt_flow=occ_flow.cpu().numpy(),
                # #     gt_pred_pcds_t=kwargs['gt_pred_pcds_t'].cpu().numpy(),gt_swin_pred_pcds_t=kwargs['gt_swin_pred_pcds_t'].cpu().numpy())
                ######################

        else:
            pred_occupancy_category =  None

        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None
        # if not self.img_backbone_only:
        #     assert len(img_metas) == 1
        if not self.pred_flow:
            pred_flow=None
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
            result_dict['pred_flow'] = pred_flow.half().cpu().numpy() if pred_flow is not None else None
            # result_dict['pred_density'] = pred_density.cpu().numpy() if pred_density is not None else None

            result_dict['pred_density'] =  None

        return bbox_list




    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        

        bbox_list = [dict() for _ in range(len(img_metas))]
        
        if  self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]
        pred_occupancy_category=None
        pred_density=None
        pred_flow=None
        # import pdb;pdb.set_trace()
        if self.with_specific_component('occupancy_head'):
            if self.test_inter_occ or self.occupancy_head.final_two_part_loss:
                # import pdb;pdb.set_trace()
                if self.occupancy_head.final_two_part_loss:
                    pred_occupancy = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)['output_voxels'][0]
                else:
                    pred_occupancy=results['inter_occs'][0]
                pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                if self.fix_void:
                    pred_occupancy = pred_occupancy[..., 1:]     

                # convert to CVPR2023 Format
                pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                pred_occupancy = torch.flip(pred_occupancy, [2])
                pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)


                pred_sem = pred_occupancy[..., :-1]
                pred_occ = pred_occupancy[..., -1:].sigmoid()
                pred_sem_category = pred_sem.argmax(-1)
                pred_free_category = (pred_occ<0.5).squeeze(-1)
                # import pdb; pdb.set_trace()
                pred_sem_category[pred_free_category] = 17
                pred_occupancy_category = pred_sem_category
            else:
                output_test = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)
                pred_occupancy = output_test['output_voxels'][0]
                if self.pred_flow:
                    # import pdb;pdb.set_trace()
                    pred_flow = output_test['output_flow'][0]
                    # pred_flow = pred_flow.permute(0, 2, 3, 4, 1)[0]
                    pred_flow = pred_flow[0]
                    pred_flow=pred_flow.permute(3, 2, 0, 1)
                    pred_flow = torch.flip(pred_flow, [2])
                    pred_flow=pred_flow[[1,0]]
                    pred_flow = torch.rot90(pred_flow, -1, [2, 3])
                    pred_flow = pred_flow.permute(2, 3, 1, 0)
                if not self.wo_pred_occ:
                
                    if self.dataset=='kitti':
                        output={}
                        output['output_voxels'] = pred_occupancy
                        output['target_voxels'] = kwargs['gt_occupancy']
                        return output
                    pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                
                    if self.fix_void:
                        pred_occupancy = pred_occupancy[..., 1:]     
                    pred_occupancy = pred_occupancy.softmax(-1)


                    # convert to CVPR2023 Format
                    pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                    pred_occupancy = torch.flip(pred_occupancy, [2])
                    pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                    pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
                    
                    if 'vis_class' in kwargs:
                        ##########
                        # import pdb;pdb.set_trace()
                        # mask=kwargs['gt_occupancy'][0]==kwargs['vis_class']
                        # mask=mask*(kwargs['gt_occupancy_ori'][0]!=255)
                        # pred_occupancy=pred_occupancy[mask[0]].mean(0)
                        
                        pred_occupancy=pred_occupancy.mean((0,1,2))

                        return pred_occupancy
                    if return_raw_occ:
                        pred_occupancy_category = pred_occupancy
                        
                        return pred_occupancy_category
                    else:
                        pred_occupancy_category = pred_occupancy.argmax(-1) 
                

                    # # do not change the order
                    # if self.occupancy_save_path is not None:
                    #     scene_name = img_metas[0]['scene_name']
                    #     sample_token = img_metas[0]['sample_idx']
                    #     mask_camera = visible_mask[0][0]
                    #     masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    #     save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', scene_name+'_'+sample_token)
                    #     np.savez_compressed(save_path, pred=masked_pred_occupancy, sample_token=sample_token) 


                    # For test server
                    if self.occupancy_save_path is not None:
                            scene_name = img_metas[0]['scene_name']
                            sample_token = img_metas[0]['sample_idx']
                            # mask_camera = visible_mask[0][0]
                            # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                            save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                            save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
                            np.savez_compressed(save_path, save_pred_occupancy.astype(np.uint8)) 

                    pred_occupancy_category= pred_occupancy_category.cpu().numpy().astype(np.uint8)
                else:
                    pred_occupancy_category =  None

        elif self.with_specific_component('maskformerocc_head'):
            if self.semantic_cluster:
                class_prototype=torch.cat((self.semantic_prototype,self.empty_embedding),dim=0)
            else:
                class_prototype=None
            # import pdb;pdb.set_trace()
            output_test = self.maskformerocc_head.simple_test(results['img_bev_feat'], img_metas=img_metas,class_prototype=class_prototype, bda=img[6],**kwargs)
            pred_occupancy = output_test['output_voxels'][0]
            if self.pred_flow:
                # import pdb;pdb.set_trace()
                pred_flow = output_test['output_flow'][0]
                pred_flow = pred_flow[0]
                pred_flow=pred_flow.permute(3, 2, 0, 1)
                pred_flow = torch.flip(pred_flow, [2])

                pred_flow = torch.rot90(pred_flow, -1, [2, 3])

                pred_flow=pred_flow[[1,0]]
                pred_flow = pred_flow.permute(2, 3, 1, 0)
            
            if not self.wo_pred_occ:
                #################
                pred_density=output_test['output_density'][0]
                
                pred_density=pred_density.unsqueeze(1)
                pred_density=pred_density.permute(0, 2, 3, 4, 1)[0]
                pred_density = pred_density.permute(3, 2, 0, 1)
                pred_density = torch.flip(pred_density, [2])
                pred_density = torch.rot90(pred_density, -1, [2, 3])
                pred_density = pred_density.permute(2, 3, 1, 0).squeeze(-1)

                ############
                if self.dataset=='kitti':
                    # import pdb;pdb.set_trace()
                    output={}
                    output['output_voxels'] = pred_occupancy
                    output['target_voxels'] = kwargs['gt_occupancy']
                    return output
                pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                
                if self.fix_void:
                    pred_occupancy = pred_occupancy[..., 1:]     
                    # import pdb;pdb.set_trace()

                # convert to CVPR2023 Format
                pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                pred_occupancy = torch.flip(pred_occupancy, [2])
                pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
                
                if 'vis_class' in kwargs:
                    ##########
                    # import pdb;pdb.set_trace()
                    mask=kwargs['gt_occupancy'][0]==kwargs['vis_class']
                    # mask=mask*(kwargs['gt_occupancy_ori'][0]!=255)
                    pred_occupancy=pred_occupancy[mask[0]].sum(0)
                    
                    # pred_occupancy=pred_occupancy.mean((0,1,2))

                    return pred_occupancy
                if return_raw_occ:
                    pred_occupancy_category = pred_occupancy
                    return pred_occupancy_category
                else:
                    pred_occupancy_category = pred_occupancy.argmax(-1) 
                    # pred_occupancy_category = pred_occupancy[...,:-1].argmax(-1) 
                    # pred_density=(pred_occupancy.argmax(-1)!=16).float()
                pred_occupancy_category = pred_occupancy_category.cpu().numpy().astype(np.uint8)

                # self.occupancy_save_path='occupancy_vis/swinbase_80088_pred2render_flow'
                if self.occupancy_save_path is not None:
                    scene_name = img_metas[0]['scene_name']
                    sample_token = img_metas[0]['sample_idx']
                    # mask_camera = visible_mask[0][0]
                    # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    # save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                    save_dir=os.path.join(self.occupancy_save_path, 'occupancy_pred',scene_name)
                    if not os.path.exists(save_dir):
                        # os.makedirs(save_dir,exists_ok=True)
                        mmcv.mkdir_or_exist(save_dir)
                    save_path = os.path.join(save_dir, f'{sample_token}.npz')
                    ##################
                    # # convert to CVPR2023 Format
                    # # import pdb;pdb.set_trace()
                    # occupancy_original = kwargs['gt_occupancy'][0][0].permute(2, 0, 1)
                    # occupancy_original = torch.flip(occupancy_original, [1])
                    # occupancy_original = torch.rot90(occupancy_original, -1, [1, 2])
                    
                    # occupancy_original = occupancy_original.permute(1, 2, 0)

                    # occ_flow = kwargs['gt_occ_flow'][0][0].permute(2, 0, 1,3)
                    # occ_flow = torch.flip(occ_flow, [1])
                    # occ_flow = torch.rot90(occ_flow, -1, [1, 2])
                    
                    # occ_flow = occ_flow.permute(1, 2, 0,3)

                    # np.savez_compressed(save_path,pred_occ= pred_occupancy_category,pred_flow=pred_flow.cpu().numpy(),gt_occ=occupancy_original.cpu().numpy(),gt_flow=occ_flow.cpu().numpy())
                   
                    np.savez_compressed(save_path,pred_occupancy_category)
                    # save_path_flow = os.path.join(save_dir, f'{sample_token}_flow.npz')
                    # np.savez_compressed(save_path_flow,pred_flow.cpu().numpy())
                # import pdb;pdb.set_trace()
           
            else:
                pred_occupancy_category =  None
                  #####################
                # self.occupancy_save_path='occupancy_vis/swinbase_80088_pred2render_flow'
                # if self.occupancy_save_path is not None:
                #     scene_name = img_metas[0]['scene_name']
                #     sample_token = img_metas[0]['sample_idx']
                #     # mask_camera = visible_mask[0][0]
                #     # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                #     # save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                #     save_dir=os.path.join(self.occupancy_save_path, 'occupancy_pred',scene_name)
                #     if not os.path.exists(save_dir):
                #         # os.makedirs(save_dir,exists_ok=True)
                #         mmcv.mkdir_or_exist(save_dir)
                #     save_path = os.path.join(save_dir, f'{sample_token}.npz')
                #     occupancy_original = kwargs['gt_occupancy'][0][0].permute(2, 0, 1)
                #     occupancy_original = torch.flip(occupancy_original, [1])
                #     occupancy_original = torch.rot90(occupancy_original, -1, [1, 2])
                    
                #     occupancy_original = occupancy_original.permute(1, 2, 0)

                #     occ_flow = kwargs['gt_occ_flow'][0][0].permute(2, 0, 1,3)
                #     occ_flow = torch.flip(occ_flow, [1])
                #     occ_flow = torch.rot90(occ_flow, -1, [1, 2])
                    
                #     occ_flow = occ_flow.permute(1, 2, 0,3)

                #     np.savez_compressed(save_path,pred_flow=pred_flow.half().cpu().numpy())
                #     # import pdb;pdb.set_trace()

                #     # np.savez_compressed(save_path,pred_flow=pred_flow.cpu().numpy(),gt_occ=occupancy_original.cpu().numpy(),gt_flow=occ_flow.cpu().numpy())

                #     # np.savez_compressed(save_path,pred_flow=pred_flow.cpu().numpy(),gt_occ=occupancy_original.cpu().numpy(),gt_flow=occ_flow.cpu().numpy(),
                # #     gt_pred_pcds_t=kwargs['gt_pred_pcds_t'].cpu().numpy(),gt_swin_pred_pcds_t=kwargs['gt_swin_pred_pcds_t'].cpu().numpy())
                ######################
        elif self.with_specific_component('voxel_head'):
            
            # target_voxels=kwargs['gt_occupancy']
            recon=results['recon']
            pred_occupancy=recon.permute(0, 2, 3, 4, 1)[0]
            ##########################
            if self.fix_void:
                pred_occupancy = pred_occupancy[..., 1:]     
            # pred_occupancy = pred_occupancy.softmax(-1)


            # convert to CVPR2023 Format
            pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
            pred_occupancy = torch.flip(pred_occupancy, [2])
            pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
            pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)

            pred_occupancy_category = pred_occupancy.argmax(-1) 
            pred_occupancy_category = pred_occupancy_category.cpu().numpy().astype(np.uint8)

        else:
            pred_occupancy_category =  None

        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None
        if not self.img_backbone_only:
            assert len(img_metas) == 1
        if not self.pred_flow:
            pred_flow=None
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
            result_dict['pred_flow'] = pred_flow.half().cpu().numpy() if pred_flow is not None else None
            # result_dict['pred_density'] = pred_density.cpu().numpy() if pred_density is not None else None

            result_dict['pred_density'] =  None

        return bbox_list


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs
