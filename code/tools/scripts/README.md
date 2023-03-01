python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT

python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT --split mini_val

python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-kitti --cam_name CAM_FRONT --split train

python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-kitti --cam_name CAM_FRONT --split val

python export_kitti.py kitti_res_to_nuscenes  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-kitti

