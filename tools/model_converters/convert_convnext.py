import torch
import copy
# 加载.pth文件
model_path = 'ckpts/convnext_base_22k_1k_384.pth'  # 更改为你的文件路径
state_dict = torch.load(model_path)#['state_dict']
state_dict=state_dict['model']
# 打印原始的键名
print("Original keys:", state_dict.keys())

# 检查并更改键名
state_dict_modified = copy.deepcopy(state_dict)
for key in state_dict.keys():

    
    if 'pwconv' in key:
        print(key,1111111111111111111)
        rename_key=copy.deepcopy(key)
        # 更改键名
        rename_key = rename_key.replace('pwconv', 'pointwise_conv')
        state_dict_modified[rename_key] = state_dict_modified.pop(key)
        print(rename_key,2222222222222222222)
    if 'dwconv' in key:
        print(key,1111111111111111111)
        rename_key=copy.deepcopy(key)
        # 更改键名
        rename_key = rename_key.replace('dwconv', 'depthwise_conv')
        state_dict_modified[rename_key] = state_dict_modified.pop(key)
        print(rename_key,2222222222222222222)
state_dict_modified2 = copy.deepcopy(state_dict_modified)        
for key in state_dict_modified.keys():
    # print(key)
    print(key,1111111111111111111)
    rename_key=copy.deepcopy(key)
    # 更改键名
    rename_key = 'img_backbone.'+rename_key
    state_dict_modified2[rename_key] = state_dict_modified2.pop(key)
    print(rename_key,2222222222222222222)
# print("Modified keys:", state_dict_modified.keys())
# import pdb; pdb.set_trace()
# 可以重复上述步骤，为其他键名做相应的更改

# 保存修改后的state_dict到一个新的.pth文件
modified_model_path = 'ckpts/convnext_base_22k_1k_384_convert.pth'  # 设定新文件的保存路径
torch.save(state_dict_modified2, modified_model_path)

# 打印修改后的键名确认更改
new_state_dict = torch.load(modified_model_path)
print("Modified keys:", new_state_dict.keys())


# import torch
# import copy
# # 加载.pth文件



# model_path1 = 'work_dirs/openocc/convnext_base_occ3d_pretrain_prototype_sup_sem_only_weight01_depth2occ_depth_ce_loss_deform_lift_offsets1_mf_head_wo_assign_length22_use_embed2_channel4_dim48_flow_l2_flow_mask_post_process_simple_conv_precess_simple_unet_occ_mask2_his16/iter_48033_ema.pth'  # 更改为你的文件路径
# state_dict1 = torch.load(model_path1)['state_dict']

# model_path2='ckpts/cascade_mask_rcnn_convnext_xlarge_22k_3x.pth'
# state_dict2 = torch.load(model_path2)

# # 打印原始的键名
# # print("Original keys:", state_dict.keys())

# # 检查并更改键名
# state_dict_modified = copy.deepcopy(state_dict1)
# for key in state_dict1.keys():
#     if 'img_backbone' in key:
#         state_dict_modified.pop(key)
# for key in state_dict2.keys():
#     if 'img_backbone'  in key:
#         state_dict_modified[key] = state_dict2[key]

# # state_dict_modified.update(state_dict2)

# # print("Modified keys:", state_dict_modified.keys())
# # import pdb; pdb.set_trace()
# # 可以重复上述步骤，为其他键名做相应的更改

# # 保存修改后的state_dict到一个新的.pth文件
# modified_model_path = 'ckpts/convnext_base_openocc_his16.pth'  # 设定新文件的保存路径
# torch.save(state_dict_modified, modified_model_path)

# # 打印修改后的键名确认更改
# new_state_dict = torch.load(modified_model_path)
# print("Modified keys:", new_state_dict.keys())
