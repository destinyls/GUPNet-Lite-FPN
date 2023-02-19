# convertion from NuScenes dataset to KITTI format
# inspired by https://github.com/poodarchu/nuscenes_to_kitti_format
# converting only camera captions (JPG files)
# converting all samples in every sequence data

# regardless of attributes indexed 2(if blocked) in KITTI label
# however, object minimum visibility level is adjustable
import os
import json
import numba
import numpy as np
from tqdm import tqdm

from nuscenes.utils import splits

start_index = 0
data_root = '/home/tsing-adept/datasets/nuScenes/'
output_root = '/home/yanglei/Datasets/nuscenes-kitti'
img_output_root = os.path.join(output_root, 'training/image_2/')
# label_output_root = os.path.join(output_root, 'training/label_2/')
label_output_root = os.path.join(output_root, 'training/label_2_attrs/')
calib_output_root = os.path.join(output_root, 'training/calib/')
imagesets_output_root = os.path.join(output_root, 'ImageSets/')

os.makedirs(img_output_root, exist_ok=True)
os.makedirs(label_output_root, exist_ok=True)
os.makedirs(calib_output_root, exist_ok=True)
os.makedirs(imagesets_output_root, exist_ok=True)

min_visibility_level = '1'
delete_dontcare_objects = True


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import numpy as np
import cv2
import os
import shutil

NameMapping = \
{
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
}

nus_attributes = \
('cycle.with_rider', 'cycle.without_rider',
     'pedestrian.moving', 'pedestrian.standing',
     'pedestrian.sitting_lying_down', 'vehicle.moving',
     'vehicle.parked', 'vehicle.stopped', 'None')

category_reflection = \
{
    'human.pedestrian.adult': 'Pedestrian',
    'human.pedestrian.child': 'Pedestrian',
    'human.pedestrian.wheelchair': 'DontCare',
    'human.pedestrian.stroller': 'DontCare',
    'human.pedestrian.personal_mobility': 'DontCare',
    'human.pedestrian.police_officer': 'Pedestrian',
    'human.pedestrian.construction_worker': 'Pedestrian',
    'animal': 'DontCare',
    'vehicle.car': 'Car',
    'vehicle.motorcycle': 'Motorcycle',
    'vehicle.bicycle': 'Cyclist',
    'vehicle.bus.bendy': 'Bus',
    'vehicle.bus.rigid': 'Bus',
    'vehicle.truck': 'Truck',
    'vehicle.construction': 'Construction_Vehicle',
    'vehicle.emergency.ambulance': 'DontCare',
    'vehicle.emergency.police': 'DontCare',
    'vehicle.trailer': 'Trailer',
    'movable_object.barrier': 'Barrier',
    'movable_object.trafficcone': 'Traffic_Cone',
    'movable_object.pushable_pullable': 'DontCare',
    'movable_object.debris': 'DontCare',
    'static_object.bicycle_rack': 'DontCare', 
}


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def calib_generation(P, calib_path):
    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P1"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P2"] = P     # Left camera transform.
    kitti_calib["P3"] = np.zeros((3, 4))  # Dummy values.
    # Cameras are already rectified.
    kitti_calib["R0_rect"] = np.eye(3)
    kitti_calib["Tr_velo_to_cam"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["Tr_imu_to_velo"] = np.zeros((3, 4))  # Dummy values.
    with open(calib_path, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))

def label_generation_attrs_velo(nusc, output_label_file, box_list):
    with open(output_label_file, 'a') as output_f:
        for box in box_list:
            # obtaining visibility level of each 3D box
            present_visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
            if present_visibility_token > min_visibility_level:
                if not (category_reflection[box.name] == 'DontCare' and delete_dontcare_objects):
                    w, l, h = box.wlh
                    x, y, z = box.center
                    yaw, pitch, roll = box.orientation.yaw_pitch_roll; yaw = -yaw
                    alpha = yaw - np.arctan2(x, z)
                    box_name = category_reflection[box.name]
                    # projecting 3D points to image plane
                    points_2d = view_points(box.corners(), cam_intrinsic, normalize=True)
                    left_2d = int(min(points_2d[0]))
                    top_2d = int(min(points_2d[1]))
                    right_2d = int(max(points_2d[0]))
                    bottom_2d = int(max(points_2d[1]))

                    ann_token = nusc.get('sample_annotation', box.token)['attribute_tokens']
                    if len(ann_token) == 0:
                        attr_name = 'None'
                    else:
                        attr_name = nusc.get('attribute', ann_token[0])['name']
                    attr_id = nus_attributes.index(attr_name)
                    line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                        box_name, 0, -1, alpha, left_2d, top_2d, right_2d, bottom_2d, h, w, l, x, y+h/2, z, yaw, attr_id)
                    output_f.write(line)

def label_generation(output_label_file, box_list):
    with open(output_label_file, 'a') as output_f:
        for box in box_list:
            # obtaining visibility level of each 3D box
            present_visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
            if present_visibility_token > min_visibility_level:
                if not (category_reflection[box.name] == 'DontCare' and delete_dontcare_objects):
                    w, l, h = box.wlh
                    x, y, z = box.center
                    yaw, pitch, roll = box.orientation.yaw_pitch_roll; yaw = -yaw
                    alpha = yaw - np.arctan2(x, z)
                    box_name = category_reflection[box.name]
                    # projecting 3D points to image plane
                    points_2d = view_points(box.corners(), cam_intrinsic, normalize=True)
                    left_2d = int(min(points_2d[0]))
                    top_2d = int(min(points_2d[1]))
                    right_2d = int(max(points_2d[0]))
                    bottom_2d = int(max(points_2d[1]))
                    line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                        box_name, 0, -1, alpha, left_2d, top_2d, right_2d, bottom_2d, h, w, l, x, y+h/2, z, yaw)
                    output_f.write(line)

if __name__ == '__main__':
    sensor_list = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
    # sensor_list = ['CAM_FRONT']
    version = 'v1.0-mini'
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    frame_counter = start_index
    '''
    if os.path.isdir(img_output_root) == True:
        print('previous image output found. deleting...')
        shutil.rmtree(img_output_root)
    os.makedirs(img_output_root)
    if os.path.isdir(label_output_root) == True:
        print('previous label output found. deleting...')
        shutil.rmtree(label_output_root)
    os.makedirs(label_output_root)
    '''
    token_dict = dict()
    dims_dict = dict()
    train_split, val_split = list(), list()
    for present_sample in tqdm(nusc.sample):
        sample_token = present_sample['token']
        scene_token = present_sample['scene_token']
        token_dict[sample_token] = list()
        # converting image data from 6 cameras (in the sensor list)
        for present_sensor in sensor_list:
            # each sensor_data corresponds to one specific image in the dataset
            sensor_data = nusc.get('sample_data', present_sample['data'][present_sensor])
            data_path, box_list, cam_intrinsic = nusc.get_sample_data(present_sample['data'][present_sensor], BoxVisibility.ALL)

            img_file = data_root + sensor_data['filename']
            seqname = str(frame_counter).zfill(6)
            if scene_token in train_scenes:
                if seqname not in train_split:
                    train_split.append(seqname)
            elif scene_token in val_scenes:
                if seqname not in val_split:
                    val_split.append(seqname)
            else:
                print("error split ...")

            token_dict[sample_token].append(seqname)
            output_label_file = label_output_root + seqname + '.txt'
            output_calib_file = calib_output_root + seqname + '.txt'

            P = np.concatenate((cam_intrinsic, np.zeros((3,1))), axis=-1)
            calib_generation(P, output_calib_file)
            if "attrs" in output_label_file:
                label_generation_attrs_velo(nusc, output_label_file, box_list)
                print("hello world")
            else:
                label_generation(output_label_file, box_list)
                
            if not os.path.getsize(output_label_file) and False:
                del_cmd = 'rm ' + output_label_file
                os.system(del_cmd)
            else:
                cmd = 'cp ' + img_file + ' ' + img_output_root + seqname + '.jpg'
                # print('copying', sensor_data['filename'], 'to', seqname + '.jpg')
                os.system(cmd)
                frame_counter += 1
            

    with open(os.path.join(output_root, 'sample_token.json'), 'w') as f:
        json.dump(token_dict, f)

    with open(os.path.join(imagesets_output_root, "train.txt"),'w') as f:
        for frame_name in train_split:
            f.write(frame_name)
            f.write("\n")
    with open(os.path.join(imagesets_output_root, "val.txt"),'w') as f:
        for frame_name in val_split:
            f.write(frame_name)
            f.write("\n")

    for box_name in dims_dict.keys():
        dim_array = np.array(dims_dict[box_name])
        print(box_name, np.mean(dim_array, axis=0))