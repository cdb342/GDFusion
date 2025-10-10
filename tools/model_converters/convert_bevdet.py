import torch
import copy
# 加载.pth文件
model_path = 'bevdet-r50-4d-stereo-cbgs.pth'  # 更改为你的文件路径
state_dict = torch.load(model_path)['state_dict']

# 打印原始的键名
print("Original keys:", state_dict.keys())

# 检查并更改键名
state_dict_modified = copy.deepcopy(state_dict)
for key in state_dict.keys():
    # print(key)
    if 'img_view_transformer.depth_net' in key:
        print(key,1111111111111111111)
        rename_key=copy.deepcopy(key)
        # 更改键名
        rename_key = rename_key.replace('img_view_transformer.depth_net', 'depth_net')
        state_dict_modified[rename_key] = state_dict_modified.pop(key)
        print(rename_key,2222222222222222222)

# print("Modified keys:", state_dict_modified.keys())
# import pdb; pdb.set_trace()
# 可以重复上述步骤，为其他键名做相应的更改

# 保存修改后的state_dict到一个新的.pth文件
modified_model_path = 'bevdet-r50-4d-stereo-cbgs_to_fbbev.pth'  # 设定新文件的保存路径
torch.save(state_dict_modified, modified_model_path)

# 打印修改后的键名确认更改
new_state_dict = torch.load(modified_model_path)
print("Modified keys:", new_state_dict.keys())
