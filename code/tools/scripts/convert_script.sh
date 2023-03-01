
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT_LEFT
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT_RIGHT
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_BACK
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_BACK_LEFT
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_BACK_RIGHT

python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT --split mini_val
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT_LEFT --split mini_val
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_FRONT_RIGHT --split mini_val
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_BACK --split mini_val
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_BACK_LEFT --split mini_val
python export_kitti.py nuscenes_gt_to_kitti  --image_count -1 --nusc_kitti_dir /root/Dataset/nuscenes-mini-kitti --cam_name CAM_BACK_RIGHT --split mini_val