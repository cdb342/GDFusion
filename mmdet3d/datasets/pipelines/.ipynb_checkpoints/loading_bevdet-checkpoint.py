# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
from .gene_depth import gene_depth
from nuscenes.nuscenes import NuScenes
import yaml
import pickle
import torch.nn.functional as F
import os

# token_map = pickle.load(open('data/nuscenes/get_lidar_token.pkl','rb'))
# nusc = NuScenes(version='v1.0-trainval',
#                         dataroot='data/nuscenes/',
#                         verbose=True)
# label_mapping_file='mmdet3d/datasets/pipelines/nuscenes.yaml'         
# with open(label_mapping_file, 'r') as stream:
#     nuscenesyaml = yaml.safe_load(stream)                        
# learning_map = nuscenesyaml['learning_map']


def coor_lidar2img(points_lidar,lidar2img,post_rots,post_trans,cid):
    points_img = points_lidar[:, :3].matmul(
        lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)

    points_img = torch.cat(
        [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
        1)

    points_img = points_img.matmul(
        post_rots[cid].T) + post_trans[cid:cid + 1, :]
    return points_img
def coor_img2lidar(coor,depth,post_trans,post_rots,lidar2img,cid):
    points_img=torch.cat([coor,depth.unsqueeze(1)],1)
    points_lidar=(points_img-post_trans[cid:cid + 1, :]).matmul(torch.inverse(post_rots[cid].T))
    points_lidar=torch.cat([points_lidar[:,:2]*points_lidar[:,2:3],points_lidar[:,2:3]],1)
    points_lidar=(points_lidar-lidar2img[:3, 3].unsqueeze(0)).matmul(torch.inverse(lidar2img[:3, :3].T))
    return points_lidar
def coor_img2lidarvox(coor,depth,post_trans,post_rots,lidar2img,cid,grid_config,lidar2lidarego):

    points_lidar=coor_img2lidar(coor,depth,post_trans,post_rots,lidar2img,cid)

    points_ego=coor_lidar2ego(points_lidar,lidar2lidarego)
    points_vox=coor_ego2lidarvox(points_ego,grid_config)

    return points_vox
def coor_lidar2ego(points_lidar,lidar2lidarego):
    return points_lidar.matmul(lidar2lidarego[:3,:3].T)+lidar2lidarego[:3,3].unsqueeze(0)

def coor_ego2lidarvox(points_lidar,grid_config):
    points_lidar[:,0]-=grid_config['x'][0]
    points_lidar[:,1]-=grid_config['y'][0]
    points_lidar[:,2]-=grid_config['z'][0]
    points_lidar=points_lidar/0.4-0.5
    points_lidar=points_lidar.round().to(torch.long)
    return points_lidar
def coor_lidar2vox(vox,sem=[None],gt_vox=False):
    valide=(vox[:,0]>=0)&(vox[:,0]<200)&(vox[:,1]>=0)\
        &(vox[:,1]<200)&(vox[:,2]>=0)&(vox[:,2]<16)
                
    vox=vox[valide]
    print(vox.shape,132123123132123132132123)
    if sem[0]:
        sem=sem[valide]
    else:
        # print(gt_vox.max(),gt_vox.min(),22222222222222222)
        sem=gt_vox[vox[:,0],vox[:,1],vox[:,2]]
        # print(sem.max(),sem.min(),444444444444444)
    
    # voxel=torch.sparse_coo_tensor(vox.t(),sem+1,torch.Size([200,200,16])).to_dense()

    output, inverse_indices,counts = torch.unique(vox,dim=0, sorted=True, return_inverse=True,return_counts=True)
    counts=counts[inverse_indices]
    
    voxel=torch.sparse_coo_tensor(vox.t(),(sem+1)/counts,(200,200,16)).to_dense().round().to(torch.long)
    voxel=voxel.numpy()
    voxel = np.where(voxel== 0, 18, voxel)
    voxel-=1
    if gt_vox is not False:
        print(voxel.max(),voxel.min(),5555555555555555)
    return voxel
@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __init__(self,down_sample_factor=1,only_mask_free=False,only_mask_nonfree=False):
        self.down_sample_factor=down_sample_factor
        self.only_mask_free=only_mask_free
        self.only_mask_nonfree=only_mask_nonfree
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        # scene_name = results['curr']['scene_name']
        # sample_token = results['curr']['token']


        # occ_gt_path = os.path.join('data/nuscenes/gts', scene_name, sample_token)
        
        # # occ_gt_path = results['curr']['occ_path']
        ##################################
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")
        
        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']
        ###########
        # occ_path2 = np.load(os.path.join('data/nuscenes/gts',results['scene_name'],results['sample_idx'],'labels.npz'))['semantics'][None]
        # print((occ_path2-semantics).sum(),111,occ_gt_path,results['scene_name'],results['sample_idx'])
        # print(results.keys(),results['curr'].keys())
        #########
        if self.only_mask_free:
            mask_camera=mask_camera | (semantics!=17)
        if self.only_mask_nonfree:
            mask_camera=mask_camera | (semantics==17)
        if self.down_sample_factor!=1:
            # print(semantics.shape,mask_lidar.shape,mask_cam)
            semantics=semantics[::self.down_sample_factor,::self.down_sample_factor,::self.down_sample_factor]
            mask_lidar=mask_lidar[::self.down_sample_factor,::self.down_sample_factor,::self.down_sample_factor]
            mask_camera=mask_camera[::self.down_sample_factor,::self.down_sample_factor,::self.down_sample_factor]

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 point_with_semantic=False):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.point_with_semantic=point_with_semantic

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        # print(pts_filename,444444444444444444444444444)
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        ##################################3
        if self.point_with_semantic:
            # lidar_sd_token = token_map[results['sample_idx']]
            # lidarseg_labels_filename = os.path.join(nusc.dataroot,
            #                                         nusc.get('lidarseg', lidar_sd_token)['filename'])

            # points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            # points_label = np.vectorize(learning_map.__getitem__)(points_label)
            save_dir='./data/lidar_seg'
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            point_name=results['pts_filename'].split('/')[-1].split('.')[0]
            save_path=os.path.join(save_dir,point_name+'.npz')
            results['points_semantics']=np.load(save_path)['lidar_seg'].flatten()
            
            # np.savez_compressed(save_path, lidar_seg=points_label)
        #############################################
            

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1,load_semantic_map=False):
        self.downsample = downsample
        self.grid_config = grid_config
        self.load_semantic_map=load_semantic_map

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        if self.load_semantic_map:
            semantics=points[:,3]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        if self.load_semantic_map:
            semantics=semantics[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        if self.load_semantic_map:
            semantics=semantics[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        if self.load_semantic_map:
            semantics=semantics[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        if self.load_semantic_map:
            semantic_map=torch.zeros((height, width), dtype=torch.float32)
            semantic_map[coor[:, 1], coor[:, 0]] = semantics
            return depth_map,semantic_map
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        semantic_map_list=[]
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            if self.load_semantic_map:
                points_semantics=results['points_semantics']
                points_img=torch.cat([points_img,torch.from_numpy(points_semantics).unsqueeze(1)],1)
            # print(points_img.shape,results['points_semantics'].shape)
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            
            if self.load_semantic_map:
                depth_map,semantic_map=depth_map
                semantic_map_list.append(semantic_map)
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        if self.load_semantic_map:
            semantic_map=torch.stack(semantic_map_list)
            results['gt_semantic_map'] = semantic_map
        return results

@PIPELINES.register_module()
class PointToMultiViewDepth_occ(object):

    def __init__(self, grid_config,depth_with_occ=False,hidden_shape=False, downsample=1,depth_expand=False,
    multi_depth=False,lidar_depth_occ_hidden=False,input_size=(256, 704),hidden_new=False,flip_later=False,SL=False,SL_size=(20,250)):
        self.downsample = downsample
        self.grid_config = grid_config
        self.depth_with_occ=depth_with_occ
        self.hidden_shape=hidden_shape
        self.depth_expand=depth_expand
        self.only_for_visual=False
        self.multi_depth=multi_depth
        self.lidar_depth_occ_hidden=lidar_depth_occ_hidden
        self.input_size=input_size
        self.frustum=self.create_frustum(grid_config['depth'], downsample).permute(1,2,0,3)
        self.hidden_new=hidden_new
        self.flip_later=flip_later
        self.grid_size=[(grid_config['x'][1]-grid_config['x'][0])//grid_config['x'][2],(grid_config['y'][1]-grid_config['y'][0])//grid_config['y'][2],(grid_config['z'][1]-grid_config['z'][0])//grid_config['z'][2]]
        self.num_dpeth=torch.arange(grid_config['depth'][0],grid_config['depth'][1],grid_config['depth'][2]).shape[0]
        if SL:
            self.input_size=SL_size
 
    def create_frustum(self, depth_cfg, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = self.input_size
        H_feat, W_feat = H_in // self.downsample, W_in // self.downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        # if self.sid:
        #     d_sid = torch.arange(self.D).float()
        #     depth_cfg_t = torch.tensor(depth_cfg).float()
        #     d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
        #                       torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
        #     d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        frustum=torch.stack((x, y, d), -1)
        

        # D x H x W x 3

        return frustum
    def points2depthmap_semantics(self, points,gt_occ, height, width,lidar2img,lidar2lidarego,post_rots,post_trans,cid,bda):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        hidden_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                points[:, 2] < self.grid_config['depth'][1]) & (
                    points[:, 2] >= self.grid_config['depth'][0])
        # depth = points[:, 2]
        # coor, depth,semantics = coor[kept1], depth[kept1],semantics[kept1]
        
        points,coor=points[kept1],coor[kept1]
####################
#expand points
        # 确定我们想要的段数
        # n_segments = 3  # 这将生成10个内部点
        if self.depth_expand:
            n_segments=self.depth_expand


            points=coor_img2lidar(coor[...,:2],points[...,2],post_trans,post_rots,lidar2img,cid)

            points=coor_lidar2ego(points,lidar2lidarego)


            half_side_length = 0.2

            

            # 生成每个维度上的坐标点，这里我们生成 n_segments + 2 个点然后移除两端的点
            lin_space_full = torch.linspace(-half_side_length, half_side_length, n_segments + 2)
            lin_space = lin_space_full[1:-1]  # 移除第一个和最后一个点以排除边界

            # 使用meshgrid来生成3D网格的坐标
            x, y, z = torch.meshgrid(lin_space, lin_space, lin_space)

            # 组合成(N, 3)的形式，其中N是网格点的数量
            grid_coordinates = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
            # print(grid_coordinates)
            points=points.unsqueeze(0).repeat(grid_coordinates.shape[0],1,1)#.reshape(-1,3)
            points=points+grid_coordinates.unsqueeze(1)
            points=points.reshape(-1,3)


            points=(points-lidar2lidarego[:3,3].unsqueeze(0)).matmul(torch.inverse(lidar2lidarego[:3,:3].T))
            points=coor_lidar2img(points,lidar2img,post_rots,post_trans,cid)

            


            semantics=semantics.unsqueeze(0).repeat(grid_coordinates.shape[0],1).reshape(-1)
            coor=torch.round(points[:, :2] / self.downsample)
            kept11 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                    points[:, 2] < self.grid_config['depth'][1]) & (
                        points[:, 2] >= self.grid_config['depth'][0])
            points,semantics,coor=points[kept11],semantics[kept11],coor[kept11]
        depth=points[:,2]
################################

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]


        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth= coor[kept2], depth[kept2]


        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        # print(self.frustum.shape,1111111111111111)
        # print(depth_map.shape,22222222222222222222222222)

        # frustum=self.frustum.reshape(-1,3)
        # frumstum_in_vox=coor_img2lidarvox(frustum[:,:2],frustum[:,2],post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
        # vox=coor_lidar2vox(frumstum_in_vox,sem=torch.ones((frumstum_in_vox.shape[0])))
        # save_dir='./vis'+'/'+str(1)
                
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, 'depth2occ.npz')
        # np.savez_compressed(save_path, voxel_frustum=vox)
        # print('saved to', save_path,3333333333333333333333333333333333333)
       

        depth_line=self.frustum[coor[:, 1], coor[:, 0]]
        # print(depth_line.shape,3333333333333333333333,depth.shape,444444444444444,depth_line.max(0,1,2),depth_line.min(0,1,2))
        num_depth,len_D,_=depth_line.shape
        depth_mask=depth_line[...,2]<depth.unsqueeze(1)+0.5
        depth_line=depth_line.reshape(-1,3)



        # frumstum_in_vox=coor_img2lidarvox(depth_line[:,:2],depth_line[:,2],post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
        frumstum_in_vox=coor_img2lidar(depth_line[:,:2],depth_line[:,2],post_trans,post_rots,lidar2img,cid)

        frumstum_in_vox=coor_lidar2ego(frumstum_in_vox,lidar2lidarego)
        frumstum_in_vox=frumstum_in_vox@bda.T
        frumstum_in_vox=coor_ego2lidarvox(frumstum_in_vox,self.grid_config)
        
        # print(bda,'asdfasdfasdf')
        
                
        gt_occ=torch.from_numpy(gt_occ)#@torch.inverse(bda.T)
        # print(frumstum_in_vox.max(0),frumstum_in_vox.min(0),555555555)
        # surface_in_vox=coor_img2lidarvox(coor,depth,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
        surface_in_vox=coor_img2lidar(coor,(depth+0.5).long().float(),post_trans,post_rots,lidar2img,cid)

        surface_in_vox=coor_lidar2ego(surface_in_vox,lidar2lidarego)
        surface_in_vox=surface_in_vox@bda.T
        surface_in_vox=coor_ego2lidarvox(surface_in_vox,self.grid_config)



        # print(surface_in_vox.max(0),surface_in_vox.min(0),6666)

        semantics=torch.ones((num_depth*len_D,))*17
        valide=(frumstum_in_vox[:,0]>=0)&(frumstum_in_vox[:,0]<self.grid_size[0])&(frumstum_in_vox[:,1]>=0)\
            &(frumstum_in_vox[:,1]<self.grid_size[1])&(frumstum_in_vox[:,2]>=0)&(frumstum_in_vox[:,2]<self.grid_size[2])
        # print(semantics.dtype,gt_occ.dtype,frumstum_in_vox.dtype,valide.dtype,777777777777777777)
        semantics[valide]=gt_occ[frumstum_in_vox[valide][:,0],frumstum_in_vox[valide][:,1],frumstum_in_vox[valide][:,2]].float()
        semantics=semantics.reshape(num_depth,len_D)


        semantics_surface=torch.ones((num_depth,))*17
        valide=(surface_in_vox[:,0]>=0)&(surface_in_vox[:,0]<self.grid_size[0])&(surface_in_vox[:,1]>=0)\
            &(surface_in_vox[:,1]<self.grid_size[1])&(surface_in_vox[:,2]>=0)&(surface_in_vox[:,2]<self.grid_size[2])
        # print(valide.shape,valide.sum(),surface_in_vox.shape,semantics_surface.shape,888888888888888888)
        semantics_surface[valide]=gt_occ[surface_in_vox[valide][:,0],surface_in_vox[valide][:,1],surface_in_vox[valide][:,2]].float()
        semantics_mask=semantics==semantics_surface.unsqueeze(1)
        # print(semantics_mask.shape,semantics_mask.max(1)[0].sum(),semantics_mask.sum(),12345123513459)
        # print((semantics_surface==17).sum(),semantics_surface.shape,(semantics_surface==18).sum(),'===========================')
        # print((semantics==17).sum(),semantics.shape,(semantics==18).sum(),'+++++++++++++++++++++')
        
        mask=~semantics_mask&(~depth_mask)
        mask=~torch.cummax(mask,dim=1)[0]
        mask=mask*(~depth_mask)
        # print(mask.max(1)[0].sum(),mask.sum(),mask.shape,depth_mask.sum(),depth_mask.shape,semantics_mask.sum(),semantics_mask.shape,depth.shape,999999999999999999)
        
        gt_depths = (depth - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        # print(gt_depths.max(),gt_depths.min(),101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010)           
        # gt_depths = (depth - self.grid_config['depth'][0] ) / \
        #                 self.grid_config['depth'][2]

        gt_depths = torch.where((gt_depths < self.num_dpeth+1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.num_dpeth+1).view(-1, self.num_dpeth+1)[:,1:]
        # print(gt_depths.sum(),(gt_depths*mask).sum(),mask.max(1)[0].sum(),(gt_depths*depth_mask).sum(),10101010110)

        mask=mask*(1-gt_depths)
        # print(mask.shape,torch.flip(mask, [1]).argmax(1).shape,'///////////////////////')
        hidden=mask.shape[1]-torch.flip(mask, [1]).argmax(1)-1
        # hidden=mask.cumsum(1).argmax(1)
        # print(hidden.shape,'........................')
        
        valide=((hidden>0)*(semantics[torch.arange(num_depth).long(),hidden]!=17)).nonzero().squeeze(1)
        d = torch.arange(*self.grid_config['depth'], dtype=torch.float)
        # print(d.max(),d.min(),depth.max(),depth.min(),2323232323232323232323231010101010)
        hidden_depth=d[hidden[valide]]
        hidden_coor=coor[valide]
        hidden_sem=semantics[valide,hidden[valide]]
        # print((hidden_sem==17).sum(),hidden_sem.shape,(hidden_sem==18).sum(),'----------------------')
        # print(hidden_sem.shape,semantics_surface.shape,456345645634536)
        hidden_map=torch.zeros((height, width), dtype=torch.float32)
        hidden_map[hidden_coor[:, 1], hidden_coor[:, 0]] = hidden_depth


        
        # for i in range(num_depth):
        #     print(depth_mask[i],gt_depths[i])
        
        if self.only_for_visual:
            return coor,depth,semantics_surface,hidden_coor,hidden_depth,hidden_sem
     
        return depth_map,hidden_map
    def points2depthmap_semantics2(self, points,semantics,gt_occ, height, width,lidar2img,lidar2lidarego,post_rots,post_trans,cid,bda):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        hidden_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                points[:, 2] < self.grid_config['depth'][1]) & (
                    points[:, 2] >= self.grid_config['depth'][0])
        # depth = points[:, 2]
        # coor, depth,semantics = coor[kept1], depth[kept1],semantics[kept1]
        
        points,coor,semantics=points[kept1],coor[kept1],semantics[kept1]
####################
#expand points
        # 确定我们想要的段数
        # n_segments = 3  # 这将生成10个内部点
        if self.depth_expand:
            n_segments=self.depth_expand


            points=coor_img2lidar(coor[...,:2],points[...,2],post_trans,post_rots,lidar2img,cid)

            points=coor_lidar2ego(points,lidar2lidarego)


            half_side_length = 0.2

            

            # 生成每个维度上的坐标点，这里我们生成 n_segments + 2 个点然后移除两端的点
            lin_space_full = torch.linspace(-half_side_length, half_side_length, n_segments + 2)
            lin_space = lin_space_full[1:-1]  # 移除第一个和最后一个点以排除边界

            # 使用meshgrid来生成3D网格的坐标
            x, y, z = torch.meshgrid(lin_space, lin_space, lin_space)

            # 组合成(N, 3)的形式，其中N是网格点的数量
            grid_coordinates = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
            # print(grid_coordinates)
            points=points.unsqueeze(0).repeat(grid_coordinates.shape[0],1,1)#.reshape(-1,3)
            points=points+grid_coordinates.unsqueeze(1)
            points=points.reshape(-1,3)


            points=(points-lidar2lidarego[:3,3].unsqueeze(0)).matmul(torch.inverse(lidar2lidarego[:3,:3].T))
            points=coor_lidar2img(points,lidar2img,post_rots,post_trans,cid)

            


            semantics=semantics.unsqueeze(0).repeat(grid_coordinates.shape[0],1).reshape(-1)
            coor=torch.round(points[:, :2] / self.downsample)
            kept11 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                    points[:, 2] < self.grid_config['depth'][1]) & (
                        points[:, 2] >= self.grid_config['depth'][0])
            points,semantics,coor=points[kept11],semantics[kept11],coor[kept11]
        depth=points[:,2]
################################

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks ,semantics= coor[sort], depth[sort], ranks[sort],semantics[sort]


        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth,semantics = coor[kept2], depth[kept2],semantics[kept2]


        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        semantics_map=torch.zeros((height, width)).long()+255
        semantics_map[coor[:, 1], coor[:, 0]] = semantics
        # print(self.frustum.shape,1111111111111111)
        # print(depth_map.shape,22222222222222222222222222)

        # frustum=self.frustum.reshape(-1,3)
        # frumstum_in_vox=coor_img2lidarvox(frustum[:,:2],frustum[:,2],post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
        # vox=coor_lidar2vox(frumstum_in_vox,sem=torch.ones((frumstum_in_vox.shape[0])))
        # save_dir='./vis'+'/'+str(1)
                
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, 'depth2occ.npz')
        # np.savez_compressed(save_path, voxel_frustum=vox)
        # print('saved to', save_path,3333333333333333333333333333333333333)
       

        depth_line=self.frustum[coor[:, 1], coor[:, 0]]
        # print(depth_line.shape,3333333333333333333333,depth.shape,444444444444444,depth_line.max(0,1,2),depth_line.min(0,1,2))
        num_depth,len_D,_=depth_line.shape
        depth_mask=depth_line[...,2]<depth.unsqueeze(1)+0.5
        depth_line=depth_line.reshape(-1,3)



        # frumstum_in_vox=coor_img2lidarvox(depth_line[:,:2],depth_line[:,2],post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
        frumstum_in_vox=coor_img2lidar(depth_line[:,:2],depth_line[:,2],post_trans,post_rots,lidar2img,cid)

        frumstum_in_vox=coor_lidar2ego(frumstum_in_vox,lidar2lidarego)
        frumstum_in_vox=frumstum_in_vox@bda.T
        frumstum_in_vox=coor_ego2lidarvox(frumstum_in_vox,self.grid_config)
        
        # print(bda,'asdfasdfasdf')
        
                
        gt_occ=torch.from_numpy(gt_occ)#@torch.inverse(bda.T)
        # print(frumstum_in_vox.max(0),frumstum_in_vox.min(0),555555555)
        # surface_in_vox=coor_img2lidarvox(coor,depth,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
        surface_in_vox=coor_img2lidar(coor,(depth+0.5).long().float(),post_trans,post_rots,lidar2img,cid)

        surface_in_vox=coor_lidar2ego(surface_in_vox,lidar2lidarego)
        surface_in_vox=surface_in_vox@bda.T
        surface_in_vox=coor_ego2lidarvox(surface_in_vox,self.grid_config)



        # print(surface_in_vox.max(0),surface_in_vox.min(0),6666)

        semantics=torch.ones((num_depth*len_D,))*17
        valide=(frumstum_in_vox[:,0]>=0)&(frumstum_in_vox[:,0]<200)&(frumstum_in_vox[:,1]>=0)\
            &(frumstum_in_vox[:,1]<200)&(frumstum_in_vox[:,2]>=0)&(frumstum_in_vox[:,2]<16)
        # print(semantics.dtype,gt_occ.dtype,frumstum_in_vox.dtype,valide.dtype,777777777777777777)
        semantics[valide]=gt_occ[frumstum_in_vox[valide][:,0],frumstum_in_vox[valide][:,1],frumstum_in_vox[valide][:,2]].float()
        semantics=semantics.reshape(num_depth,len_D)


        semantics_surface=torch.ones((num_depth,))*17
        valide=(surface_in_vox[:,0]>=0)&(surface_in_vox[:,0]<200)&(surface_in_vox[:,1]>=0)\
            &(surface_in_vox[:,1]<200)&(surface_in_vox[:,2]>=0)&(surface_in_vox[:,2]<16)
        # print(valide.shape,valide.sum(),surface_in_vox.shape,semantics_surface.shape,888888888888888888)
        semantics_surface[valide]=gt_occ[surface_in_vox[valide][:,0],surface_in_vox[valide][:,1],surface_in_vox[valide][:,2]].float()
        semantics_mask=semantics==semantics_surface.unsqueeze(1)
        # print(semantics_mask.shape,semantics_mask.max(1)[0].sum(),semantics_mask.sum(),12345123513459)
        # print((semantics_surface==17).sum(),semantics_surface.shape,(semantics_surface==18).sum(),'===========================')
        # print((semantics==17).sum(),semantics.shape,(semantics==18).sum(),'+++++++++++++++++++++')
        
        mask=~semantics_mask&(~depth_mask)
        mask=~torch.cummax(mask,dim=1)[0]
        mask=mask*(~depth_mask)
        # print(mask.max(1)[0].sum(),mask.sum(),mask.shape,depth_mask.sum(),depth_mask.shape,semantics_mask.sum(),semantics_mask.shape,depth.shape,999999999999999999)
        
        gt_depths = (depth - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        # print(gt_depths.max(),gt_depths.min(),101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010)           
        # gt_depths = (depth - self.grid_config['depth'][0] ) / \
        #                 self.grid_config['depth'][2]

        gt_depths = torch.where((gt_depths < self.num_dpeth+1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.num_dpeth+1).view(-1, self.num_dpeth+1)[:,1:]
        # print(gt_depths.sum(),(gt_depths*mask).sum(),mask.max(1)[0].sum(),(gt_depths*depth_mask).sum(),10101010110)

        mask=mask*(1-gt_depths)
        # print(mask.shape,torch.flip(mask, [1]).argmax(1).shape,'///////////////////////')
        hidden=mask.shape[1]-torch.flip(mask, [1]).argmax(1)-1
        # hidden=mask.cumsum(1).argmax(1)
        # print(hidden.shape,'........................')
        
        valide=((hidden>0)*(semantics[torch.arange(num_depth).long(),hidden]!=17)).nonzero().squeeze(1)
        d = torch.arange(*self.grid_config['depth'], dtype=torch.float)
        # print(d.max(),d.min(),depth.max(),depth.min(),2323232323232323232323231010101010)
        hidden_depth=d[hidden[valide]]
        hidden_coor=coor[valide]
        hidden_sem=semantics[valide,hidden[valide]]
        # print((hidden_sem==17).sum(),hidden_sem.shape,(hidden_sem==18).sum(),'----------------------')
        # print(hidden_sem.shape,semantics_surface.shape,456345645634536)
        hidden_map=torch.zeros((height, width), dtype=torch.float32)
        hidden_map[hidden_coor[:, 1], hidden_coor[:, 0]] = hidden_depth


        
        # for i in range(num_depth):
        #     print(depth_mask[i],gt_depths[i])
        
        if self.only_for_visual:
            return coor,depth,semantics_surface,hidden_coor,hidden_depth,hidden_sem
     
        return depth_map,hidden_map,semantics_map



    def points2depthmap(self, points, height, width,lidar2img,lidar2lidarego,post_rots,post_trans,cid):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        hidden_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                points[:, 2] < self.grid_config['depth'][1]) & (
                    points[:, 2] >= self.grid_config['depth'][0])
        # depth = points[:, 2]
        # coor, depth,semantics = coor[kept1], depth[kept1],semantics[kept1]

        points,coor=points[kept1],coor[kept1]
####################
#expand points
        # 确定我们想要的段数
        # n_segments = 3  # 这将生成10个内部点
        if self.depth_expand:
            n_segments=self.depth_expand


            points=coor_img2lidar(coor[...,:2],points[...,2],post_trans,post_rots,lidar2img,cid)

            points=coor_lidar2ego(points,lidar2lidarego)


            half_side_length = 0.2

            

            # 生成每个维度上的坐标点，这里我们生成 n_segments + 2 个点然后移除两端的点
            lin_space_full = torch.linspace(-half_side_length, half_side_length, n_segments + 2)
            lin_space = lin_space_full[1:-1]  # 移除第一个和最后一个点以排除边界

            # 使用meshgrid来生成3D网格的坐标
            x, y, z = torch.meshgrid(lin_space, lin_space, lin_space)

            # 组合成(N, 3)的形式，其中N是网格点的数量
            grid_coordinates = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
            # print(grid_coordinates)
            points=points.unsqueeze(0).repeat(grid_coordinates.shape[0],1,1)#.reshape(-1,3)
            points=points+grid_coordinates.unsqueeze(1)
            points=points.reshape(-1,3)


            points=(points-lidar2lidarego[:3,3].unsqueeze(0)).matmul(torch.inverse(lidar2lidarego[:3,:3].T))
            points=coor_lidar2img(points,lidar2img,post_rots,post_trans,cid)

            


            semantics=semantics.unsqueeze(0).repeat(grid_coordinates.shape[0],1).reshape(-1)
            coor=torch.round(points[:, :2] / self.downsample)
            kept11 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                    points[:, 2] < self.grid_config['depth'][1]) & (
                        points[:, 2] >= self.grid_config['depth'][0])
            points,semantics,coor=points[kept11],semantics[kept11],coor[kept11]
        depth=points[:,2]
################################

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]


        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]


        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        if self.only_for_visual:
            return coor,depth
     
        return depth_map

    def points2depthmap_occ(self, points,semantics, height, width,lidar2img,lidar2lidarego,post_rots,post_trans,cid):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        hidden_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)

        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                points[:, 2] < self.grid_config['depth'][1]) & (
                    points[:, 2] >= self.grid_config['depth'][0])
        # depth = points[:, 2]
        # coor, depth,semantics = coor[kept1], depth[kept1],semantics[kept1]

        points,semantics,coor=points[kept1],semantics[kept1],coor[kept1]
####################
#expand points
        # 确定我们想要的段数
        # n_segments = 3  # 这将生成10个内部点
        if self.depth_expand:
            n_segments=self.depth_expand


            points=coor_img2lidar(coor[...,:2],points[...,2],post_trans,post_rots,lidar2img,cid)

            points=coor_lidar2ego(points,lidar2lidarego)


            half_side_length = 0.2

            

            # 生成每个维度上的坐标点，这里我们生成 n_segments + 2 个点然后移除两端的点
            lin_space_full = torch.linspace(-half_side_length, half_side_length, n_segments + 2)
            lin_space = lin_space_full[1:-1]  # 移除第一个和最后一个点以排除边界

            # 使用meshgrid来生成3D网格的坐标
            x, y, z = torch.meshgrid(lin_space, lin_space, lin_space)

            # 组合成(N, 3)的形式，其中N是网格点的数量
            grid_coordinates = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
            # print(grid_coordinates)
            points=points.unsqueeze(0).repeat(grid_coordinates.shape[0],1,1)#.reshape(-1,3)
            points=points+grid_coordinates.unsqueeze(1)
            points=points.reshape(-1,3)


            points=(points-lidar2lidarego[:3,3].unsqueeze(0)).matmul(torch.inverse(lidar2lidarego[:3,:3].T))
            points=coor_lidar2img(points,lidar2img,post_rots,post_trans,cid)

            


            semantics=semantics.unsqueeze(0).repeat(grid_coordinates.shape[0],1).reshape(-1)
            coor=torch.round(points[:, :2] / self.downsample)
            kept11 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                    points[:, 2] < self.grid_config['depth'][1]) & (
                        points[:, 2] >= self.grid_config['depth'][0])
            points,semantics,coor=points[kept11],semantics[kept11],coor[kept11]
        depth=points[:,2]
################################
        
        coor_surface,depth_surface,semantics_surface,coor_hidden,depth_hidden,semantics_hidden=gene_depth(depth,semantics,coor,0.51,width)


        coor_surface = coor_surface.to(torch.long)
        depth_map[coor_surface[:, 1], coor_surface[:, 0]] = depth_surface
     
        coor_hidden = coor_hidden.to(torch.long)
        hidden_map[coor_hidden[:, 1], coor_hidden[:, 0]] = depth_hidden

        # print(len(depth_surface),len(depth_hidden),999999999,depth_hidden.max(),depth_hidden.min(),
        #     torch.unique(coor_hidden,dim=0).shape,torch.unique(coor_surface,dim=0).shape,coor_hidden.max(),coor_hidden.min(),coor_surface.max(),coor_surface.min())
            
        # valid_mask=(hidden_map>0)
        
        # valid_mask=valid_mask&valid_mask2
        # zz=hidden_map[valid_mask]-depth_map[valid_mask]
        # print((zz<0).sum(),666666666666666666)
#####################################
#测试hideen map可视化
        # depth_semantic_map = torch.zeros((height, width), dtype=torch.float32)
        # depth_semantic_map[coor_surface[:, 1], coor_surface[:, 0]] = semantics_surface.float()
        
        # hidden_semantic_map = torch.zeros((height, width), dtype=torch.float32)
        # hidden_semantic_map[coor_hidden[:, 1], coor_hidden[:, 0]] = semantics_hidden.float()

        # # print((hidden_map>0).sum(),7777777777777)
        # # invalid_mask2=hidden_map-depth_map<-0.
        # # hidden_map[invalid_mask2]=0
        # # print((invalid_mask2).sum(),88888888888888)
        # # print()

        # sparse_hidden = hidden_map.to_sparse()
        # coor_hidden_,depth_hidden_ = sparse_hidden.indices().t(), sparse_hidden.values()
        # print(abs(coor_hidden_[depth_hidden_.sort()[1]]-coor_hidden[depth_hidden.sort()[1]]).sum(),11132164163,coor_hidden.max(),
        # coor_hidden.min(),coor_hidden_.max(),coor_hidden_.min(),coor_hidden.dtype,coor_hidden_.dtype,
        # abs(coor_hidden_.sort()[0]-coor_hidden.sort()[0]).sum(),99923421342134234)
        # print(abs(depth_hidden_.sort()[0]-depth_hidden.sort()[0]).sum(),99923421342134234)


        # semantics_hidden_=hidden_semantic_map[coor_hidden_[:, 0], coor_hidden_[:, 1]]
        # depth_hidden_=depth_map[coor_hidden_[:, 0], coor_hidden_[:, 1]]
        # print(abs(semantics_hidden_.sort()[0]-semantics_hidden.sort()[0]).sum(),555555555555555555)
        # print(abs(torch.cat([coor_hidden_,depth_hidden_.unsqueeze(1),semantics_hidden_.unsqueeze(1)],1).sort()[0]-
        # torch.cat([coor_hidden,depth_hidden.unsqueeze(1),semantics_hidden.unsqueeze(1)],1).sort()[0]).sum(),555777777777777777777555)

        # print(semantics_hidden_.dtype,semantics_hidden.dtype,depth_hidden_.dtype,depth_hidden.dtype,564564646)
        # z=torch.cat([coor_hidden_,depth_hidden_.unsqueeze(1),semantics_hidden_.unsqueeze(1)],1).sort()[0]

        # zz=torch.cat([coor_hidden,depth_hidden.unsqueeze(1),semantics_hidden.unsqueeze(1)],1)#.sort()[0]
        # indices_ = torch.randperm(zz.size(0))
        # print(indices_.shape,2342353452)
        # # indices=zz.argsort(dim=0)
        # # indices=zz=torch.cat([coor_hidden,depth_hidden.unsqueeze(1),semantics_hidden.unsqueeze(1)],1).sort()[1]
        # # print(indices.shape,453245645756,(indices.sum(),indices_.sum()))
        # # zz=zz[indices]
        # # coor_hidden_=zz[:,:2]
        # # depth_hidden_=zz[:,2]
        # # semantics_hidden_=zz[:,3]
        if self.only_for_visual:
            return coor_surface,depth_surface,semantics_surface,coor_hidden,depth_hidden,semantics_hidden.float()

##########################################
        return depth_map,hidden_map

    def __call__(self, results):
        
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        hidden_map_list = []
        semantics_map_list = []
        ####################
        vox_surface_list=[]
        vox_hidden_list=[]
        vox_list=[]
        vox_lidar_hidden_list=[]
        sem_surface_list=[]
        sem_hidden_list=[]
        sem_list=[]
        sem_lidar_hidden_list=[]
        points_lidar_list=[]
        frustum_list=[]
        ###################
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)


            if not self.depth_with_occ:
                points_lidar = results['points']
                points_lidar=points_lidar.tensor
                # if self.hidden_shape:
                #     semantics=results['points_semantics']
                #     semantics=torch.from_numpy(semantics)
                points_img=coor_lidar2img(points_lidar,lidar2img,post_rots,post_trans,cid)


            else:
                occ=results['voxel_semantics']+1
                occ[results['mask_camera']]=18
                occ = np.where(occ== 18, 0, occ)
                # occ=np.where(occ== 256, 1, occ)
                occ=torch.from_numpy(occ)
                sparse_occ = occ.to_sparse()
                points_voxel,semantics = sparse_occ.indices().t(), sparse_occ.values()
                # print(points_voxel.shape,2222222222)
                semantics_occ=semantics-1
                # print(points_voxel.shape,semantics.shape,     111111111)
                points=points_voxel.float()
                voxel_size=[0.4,0.4,0.4]
                points=(points+0.5) * torch.tensor(voxel_size)
                tem=points.clone()
                points[:,0]+=self.grid_config['x'][0]
                points[:,1]+=self.grid_config['y'][0]
                points[:,2]+=self.grid_config['z'][0]
                points=points.matmul(torch.inverse(bda.T))
                points_lidar_occ=(points-lidar2lidarego[:3,3].unsqueeze(0)).matmul(torch.inverse(lidar2lidarego[:3,:3].T))
                points_img=coor_lidar2img(points_lidar_occ,lidar2img,post_rots,post_trans,cid)
                # points_lidar=points_lidar_occ
                if self.multi_depth:
                    num_occ_depth=len(points_lidar_occ)
                    points_lidar = results['points']
                    points_lidar=points_lidar.tensor[...,:3]
                    points_lidar=torch.cat([points_lidar_occ,points_lidar],0)
                    semantics=results['points_semantics']
                    semantics=torch.from_numpy(semantics)
                    semantics=torch.cat([semantics_occ,semantics],0)
                    points_img=coor_lidar2img(points_lidar,lidar2img,post_rots,post_trans,cid)


###############################
            
            
###############################
            

            ######################
           # test
            # points_img=(points_img-post_trans[cid:cid + 1, :]).matmul(torch.inverse(post_rots[cid].T))
            
            # points_img=torch.cat([points_img[:,:2]*points_img[:,2:3],points_img[:,2:3]],1)
            
            # points_img=(points_img-lidar2img[:3, 3].unsqueeze(0)).matmul(torch.inverse(lidar2img[:3, :3].T))
            
            # points_img[:,0]-=self.grid_config['x'][0]
            # points_img[:,1]-=self.grid_config['y'][0]
            # points_img[:,2]-=self.grid_config['z'][0]
            # print(abs(points_img-tem).sum(),8888888888888888)
            # points_img=points_img/0.4-0.5
            # points_img=points_img.round().to(torch.long)
            # print((points_voxel-points_img).sum(),99999999999999999)
            ######################
            ###################
            #可视化
            # self.only_for_visual=True
            if self.only_for_visual:
                #voxel_depth
                # coor_surface,depth_surface,semantics_surface,coor_hidden,depth_hidden,semantics_hidden = \
                #         self.points2depthmap_occ(points_img[:num_occ_depth],semantics[:num_occ_depth], imgs.shape[2],
                #                                     imgs.shape[3],lidar2img,lidar2lidarego ,post_rots,post_trans,cid)   
                # # depth_map_list.append(depth_map)   
                # # hidden_map_list.append(hidden_map) 
                # # 
                # voxel_surface=coor_img2lidarvox(coor_surface,depth_surface,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)   
                # voxel_hidden=coor_img2lidarvox(coor_hidden,depth_hidden,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)   

                # vox_surface_list.append(voxel_surface)
                # vox_hidden_list.append(voxel_hidden)
                # sem_surface_list.append(semantics_surface)
                # sem_hidden_list.append(semantics_hidden)
                # ##############################
                # #lidar_depth
                # points_lidar_lidar = results['points']
                # points_lidar_lidar=points_lidar_lidar.tensor
                # semantics_lidar=torch.from_numpy(results['points_semantics'])
                # points_img_lidar=coor_lidar2img(points_lidar_lidar,lidar2img,post_rots,post_trans,cid)

                # coor,depth,semantics_lidar,coor_lidar_hidden,depth_lidar_hidden,semantics_lidar_hidden=\
                #     self.points2depthmap_occ(points_img_lidar,semantics_lidar, imgs.shape[2],
                #                     imgs.shape[3],lidar2img,lidar2lidarego ,post_rots,post_trans,cid) 

                # # coor,depth=self.points2depthmap(points_img_lidar,imgs.shape[2],
                # #                     imgs.shape[3],lidar2img,post_rots,post_trans,cid)         
                # vox=coor_img2lidarvox(coor,depth,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
                # vox_lidar_hidden=coor_img2lidarvox(coor_lidar_hidden,depth_lidar_hidden,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)
                # vox_list.append(vox)
                # sem_list.append(semantics_lidar)
                # sem_lidar_hidden_list.append(semantics_lidar_hidden)
                # vox_lidar_hidden_list.append(vox_lidar_hidden)
                ###############################
                coor_surface,depth_surface,semantics_surface,coor_hidden,depth_hidden,semantics_hidden = self.points2depthmap_semantics(points_img[num_occ_depth:],semantics[num_occ_depth:],results['voxel_semantics'], imgs.shape[2],imgs.shape[3],
                            lidar2img,lidar2lidarego,post_rots,post_trans,cid,bda)
                voxel_surface=coor_img2lidarvox(coor_surface,depth_surface,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)   
                voxel_hidden=coor_img2lidarvox(coor_hidden,depth_hidden,post_trans,post_rots,lidar2img,cid,self.grid_config,lidar2lidarego)               
                vox_surface_list.append(voxel_surface)
                vox_hidden_list.append(voxel_hidden)
                sem_surface_list.append(semantics_surface)
                sem_hidden_list.append(semantics_hidden)            
                ##############################
            else:   
                
                if self.hidden_shape:
                    if self.hidden_new:
                        # depth_map,hidden_map,semantics_map= self.points2depthmap_semantics(points_img[num_occ_depth:],semantics[num_occ_depth:],results['voxel_semantics'], imgs.shape[2],imgs.shape[3],
                        #     lidar2img,lidar2lidarego,post_rots,post_trans,cid,bda)
                        if self.flip_later:
                            gt_occ=results['voxel_semantics_ori']
                        else:
                            gt_occ=results['voxel_semantics']
                        
                        depth_map,hidden_map= self.points2depthmap_semantics(points_img,gt_occ, self.input_size[0],self.input_size[1],
                            lidar2img,lidar2lidarego,post_rots,post_trans,cid,bda)    

                    else:
                        if self.multi_depth:
                            if self.lidar_depth_occ_hidden:
                                depth_map = self.points2depthmap(points_img[num_occ_depth:], imgs.shape[2],
                                                imgs.shape[3],lidar2img,lidar2lidarego,post_rots,post_trans,cid)  
                            else:                      
                                depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                                imgs.shape[3],lidar2img,lidar2lidarego,post_rots,post_trans,cid)    
                            _,hidden_map = self.points2depthmap_occ(points_img[:num_occ_depth],semantics[:num_occ_depth], imgs.shape[2],
                                                        imgs.shape[3],lidar2img,lidar2lidarego ,post_rots,post_trans,cid)  
                                                                        
                        else:
                            depth_map,hidden_map = self.points2depthmap_occ(points_img,semantics_occ, imgs.shape[2],
                                                            imgs.shape[3],lidar2img,lidar2lidarego ,post_rots,post_trans,cid)  
                            # depth_map,hidden_map = self.points2depthmap(points_img, imgs.shape[2],
                            #                                 imgs.shape[3],lidar2img,post_rots,post_trans,cid)                                  
                                                    
                    depth_map_list.append(depth_map)   
                    hidden_map_list.append(hidden_map)
                    # semantics_map_list.append(semantics_map)
                                            
                else:
                    if self.lidar_depth_occ_hidden:
                        depth_map = self.points2depthmap(points_img[num_occ_depth:], imgs.shape[2],
                                                imgs.shape[3],lidar2img,lidar2lidarego,post_rots,post_trans,cid)   
                    else:
                        depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                                imgs.shape[3],lidar2img,lidar2lidarego,post_rots,post_trans,cid)  
                    depth_map_list.append(depth_map)
        if self.only_for_visual:
            ##########################
            #visualize
            
            vox_hidden=torch.cat(vox_hidden_list,0)
            vox_surface=torch.cat(vox_surface_list,0)
            # vox_surface=vox_surface.matmul(lidar2lidarego[:3,:3].T)+lidar2lidarego[:3,3].unsqueeze(0)
            # vox_hidden=vox_hidden.matmul(lidar2lidarego[:3,:3].T)+lidar2lidarego[:3,3].unsqueeze(0)


            sem_hidden=torch.cat(sem_hidden_list,0)
            sem_surface=torch.cat(sem_surface_list,0)
            

            # vox_from_lidar=torch.cat(vox_list,0)
            # sem=torch.cat(sem_list,0)

            # sem_lidar_hidden=torch.cat(sem_lidar_hidden_list,0)
            # vox_lidar_hidden=torch.cat(vox_lidar_hidden_list,0)


            # print(vox_from_lidar.max(),vox_from_lidar.min(),555555555555555555)
            # vox_from_lidar = results['points'][...,:3].tensor
            # vox_from_lidar=torch.cat(points_lidar_list,0)
            # vox_from_lidar=vox_from_lidar.matmul(lidar2lidarego[:3,:3].T)+lidar2lidarego[:3,3].unsqueeze(0)

           
            # vox_from_lidar=coor_ego2lidarvox(vox_from_lidar,self.grid_config)
            # print(vox_from_lidar.max(dim=0),vox_from_lidar.min(dim=0),111111111111111111111)

            print('num_surface',len(vox_surface),'num_hidden',len(vox_hidden),99999999999999)
        
            print(sem_hidden.shape,sem_surface.shape,(sem_hidden==17).sum(),(sem_surface==17).sum(),4444444444444)
            voxel_hidden=coor_lidar2vox(vox_hidden,sem_hidden)
            voxel_surface=coor_lidar2vox(vox_surface,sem_surface)
            # voxel_from_lidar=coor_lidar2vox(vox_from_lidar,sem)
            # voxel_lidar_hidden=coor_lidar2vox(vox_lidar_hidden,sem_lidar_hidden)
            # voxel_from_lidar=coor_lidar2vox(vox_from_lidar,[None],torch.from_numpy(results['voxel_semantics']))
            # voxel_from_lidar=vox_from_lidar
            # print(voxel_hidden.max(),voxel_hidden.min(),233333333333333333333333333333)

     
            # print(voxel_hidden.max(),voxel_hidden.min(),voxel_surface.max(),voxel_surface.min(),2222222222222222222222)

            save_dir='./vis'+'/'+str(results['sample_idx'])
                
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'depth2occ.npz')
            # np.savez_compressed(save_path, voxel_surface=voxel_surface,voxel_hidden=voxel_hidden,
            #     voxel_from_lidar=voxel_from_lidar,voxel_gt=results['voxel_semantics'],voxel_lidar_hidden=voxel_lidar_hidden)
            np.savez_compressed(save_path, voxel_surface=voxel_surface,voxel_hidden=voxel_hidden,
                voxel_gt=results['voxel_semantics'])    
            print('saved to', save_path)

            #$#######################
        else:
            # vox=np.stack(frustum_list,0).sum(0)
            # save_dir='./vis'+'/'+str(1)
                
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # save_path = os.path.join(save_dir, 'depth2occ.npz')
            # np.savez_compressed(save_path, voxel_frustum=vox)
            # print('saved to', save_path,3333333333333333333333333333333333333)


            if self.hidden_shape:
                
                depth_map = torch.stack(depth_map_list)
                hidden_map = torch.stack(hidden_map_list)
                results['gt_depth'] = depth_map
                results['gt_hidden'] = hidden_map
                # semantics_map = torch.stack(semantics_map_list)
                # results['gt_depth_semantics'] = semantics_map
                
            else:
                depth_map = torch.stack(depth_map_list)

                results['gt_depth'] = depth_map

        return results

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        if self.is_train:                         
            results['voxel_semantics_ori']=results['voxel_semantics'].copy()                         
        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
        return results
