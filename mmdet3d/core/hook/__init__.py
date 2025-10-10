# Copyright (c) OpenMMLab. All rights reserved.
from .ema import MEGVIIEMAHook
from .utils import is_parallel
from .sequentialsontrol import SequentialControlHook
from .sequentialsontrol_multi import SequentialControlHookMulti
from .sequentialsontrol_flow import SequentialControlHookFlow
from .weightcontrol import WeightControlHook
from .syncbncontrol import SyncbnControlHook
from .fusionweightcontrol import FusionRateControlHook
from .fusionweightcontrol_depth import FusionRateControlDepthHook
from .fusionweightcontrol_pose import FusionRateControlPoseHook
__all__ = ['MEGVIIEMAHook', 'is_parallel', 'SequentialControlHook', 'SequentialControlHookMulti', 'WeightControlHook', 'SequentialControlHookFlow',
           'SyncbnControlHook','FusionRateControlHook','FusionRateControlDepthHook','FusionRateControlPoseHook']
