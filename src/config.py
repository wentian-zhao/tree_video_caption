import os
import sys

data_path = os.path.join('..', 'data')
video_feat_path = os.path.join(data_path, 'video_feat')

java_path = '/usr/local/lib/jdk1.8.0_241/bin/java'
os.environ['PATH'] += ':' + os.path.split(java_path)[0]



coco_ann_path = '/media/wentian/sdb1/caption_datasets/caption_datasets/dataset_coco.json'
flickr30k_ann_path = '/media/wentian/sdb1/caption_datasets/caption_datasets/dataset_flickr30k.json'

coco_feat_path = '/media/wentian/sda7/feat_sdc1/caption_features/coco/coco_fc.h5'
coco_feat_att_path = '/media/wentian/sda7/feat_sdc1/caption_features/coco/coco_att.h5'

# bottom up feature
# coco_feat_path = '/media/wentian/nvme1/bottom_up_feat/coco/fixed/coco_fc.h5'
# coco_feat_att_path = '/media/wentian/nvme1/bottom_up_feat/coco/fixed/coco_att.h5'
# coco_feat_path = '/media/wentian/nvme1/bottom_up_feat/coco/adapt/coco_fc.h5'
# coco_feat_att_path = '/media/wentian/nvme1/bottom_up_feat/coco/adapt/coco_att.h5'

# coco_feat_dir = '/media/wentian/nvme1/bottom_up_feat/coco/adapt/cocobu/cocobu_fc/'
# coco_feat_att_dir = '/media/wentian/nvme1/bottom_up_feat/coco/adapt/cocobu/cocobu_att/'

flickr30k_feat_dir = r'/media/wentian/nvme1/bottom_up_feat/SCAN/data_npy'

coco_dep_dir = r'/media/wentian/sdb1/work/graph_cap_2/data/coco/dep_tree_coco.json'
flickr30k_dep_dir = r'/media/wentian/sdb1/work/graph_cap_2/data/flickr30k/dep_tree_flickr30k.json'

msvd_frame_feat_path = r'/media/wentian/sda7/RMN_video_feat/MSVD/npy_frame'
msvd_region_feat_path = r'/media/wentian/sda7/RMN_video_feat/MSVD/npy_region'

msrvtt_frame_feat_path = r'/media/wentian/sda7/RMN_video_feat/MSR-VTT/npy_frame'
msrvtt_region_feat_path = r'/media/wentian/sda7/RMN_video_feat/MSR-VTT/npy_region'

charades_feat_dir = r'/media/wentian/sda7/charades_feature_resnet152'

activitynet_c3d_feat_dir = r'/media/wentian/sdb1/caption_datasets/activitynet captions/c3d/activitynet_v1.3/npy'
activitynet_references = [
    '/media/wentian/sdb1/caption_datasets/activitynet captions/captions/val_1.json',
    '/media/wentian/sdb1/caption_datasets/activitynet captions/captions/val_2.json',
]
_activitynet_resnet_bn_feat_dir = r'/media/wentian/nvme1/densecap_video_feat/activitynet_resnet_bn'
activitynet_resnet_bn_feat_dir = [
    os.path.join(_activitynet_resnet_bn_feat_dir, 'training'), os.path.join(_activitynet_resnet_bn_feat_dir, 'validation')
]