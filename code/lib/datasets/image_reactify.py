import cv2
import math

import numpy as np
from mmdet.datasets.builder import PIPELINES
from lib.datasets.visual_utils import compute_box_3d_camera, project_to_image, draw_box_3d

@PIPELINES.register_module()
class ImageReactify(object):
    def __init__(self, ratio_range, roll_range, pitch_range):
        self.roll_range = roll_range
        self.pitch_range = pitch_range
        self.ratio_range = ratio_range

    def equation_plane(self, points): 
        x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
        x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
        x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * x1 - b * y1 - c * z1)
        return [a, b, c, d]
    
    def parse_roll_pitch(self, lidar2cam):
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
        ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
        denorm = self.equation_plane(ground_points_cam)
        
        origin_vector = np.array([0, 1.0, 0])
        target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
        target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
        target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
        target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
        roll = math.acos(np.inner(origin_vector, target_vector_xy))
        pitch = math.acos(np.inner(origin_vector, target_vector_yz))
        roll = -1 * self.rad2degree(roll) if target_vector_xy[0] > 0 else self.rad2degree(roll)
        pitch = -1 * self.rad2degree(pitch) if target_vector_yz[1] > 0 else self.rad2degree(pitch)
        return roll, pitch
    
    def get_denorm(self, src_denorm_file):
        with open(src_denorm_file, 'r') as f:
            lines = f.readlines()
        denorm = np.array([float(item) for item in lines[0].split(' ')])
        return denorm
    
    def reactify_roll_params(self, image, lidar2cam, cam_intrinsic):        
        roll = np.random.uniform(self.roll_range[0], self.roll_range[1])
        roll_rad = self.degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_roll, lidar2cam)
        lidar2img_rectify = (cam_intrinsic @ lidar2cam_rectify)
        h, w, _ = image.shape
        center = (int(cam_intrinsic[0, 2]), int(cam_intrinsic[1, 2]))
        M = cv2.getRotationMatrix2D(center, -1 * self.rad2degree(roll_rad), 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return lidar2cam_rectify, lidar2img_rectify, image
    
    def reactify_pitch_params(self, image, lidar2cam, cam_intrinsic):
        pitch = np.random.uniform(self.pitch_range[0], self.pitch_range[1])            
        pitch = self.degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch), -math.sin(pitch), 0], 
                                  [0,math.sin(pitch), math.cos(pitch), 0],
                                  [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_pitch, lidar2cam)
        lidar2img_rectify = (cam_intrinsic @ lidar2cam_rectify)

        M = self.get_M(lidar2cam[:3,:3], cam_intrinsic[:3,:3], lidar2cam_rectify[:3,:3], cam_intrinsic[:3,:3])
        center = cam_intrinsic[:2, 2]  # w, h
        center_ref = np.array([center[0], center[1], 1.0])
        center_ref = np.matmul(M, center_ref.T)[:2]
        transform_pitch = int(center_ref[1] - center[1])
        translation_matrix = np.array([
            [1, 0, 0],
            [0, 1, transform_pitch]
        ], dtype=np.float32)
        h, w, _ = image.shape
        image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(w, h))
        return lidar2cam_rectify, lidar2img_rectify, image
    
    def reactify_ratio_params(self, image, cam_intrinsic):
        ratio = np.random.uniform(self.ratio_range[0], self.ratio_range[1])
        cam_intrinsic_rectify = cam_intrinsic.copy()
        cam_intrinsic_rectify[:2,:2] = cam_intrinsic[:2,:2] * ratio
        
        center = cam_intrinsic[:2, 2].astype(np.int32) 
        center = (int(center[0]), int(center[1]))
        H, W, _ = image.shape
        new_W, new_H = (int(W * ratio), int(H * ratio))    
        img = cv2.resize(image, (new_W, new_H), cv2.INTER_AREA)
        h_min = int(center[1] * abs(1.0 - ratio))
        w_min = int(center[0] * abs(1.0 - ratio))
        
        if ratio <= 1.0:
            image = np.zeros_like(image)
            image[h_min:h_min + new_H, w_min:w_min + new_W, :] = img     
        else:
            image = img[h_min:h_min + H, w_min:w_min + W, :]
        return cam_intrinsic_rectify, image
            
    def rad2degree(self, radian):
        return radian * 180 / np.pi
    
    def degree2rad(self, degree):
        return degree * np.pi / 180

    def visualize(self, img, gt_bboxes_3d, cam2img, c=(0, 255, 0)):
        print(img.shape, img.dtype)
        print(gt_bboxes_3d.tensor.shape)
        print(np.array(cam2img).shape)
        
        gt_bboxes_3d = gt_bboxes_3d.tensor.numpy()
        print(gt_bboxes_3d.shape)

        cam2img = np.array(cam2img)
        P = np.concatenate(
            [cam2img, np.zeros((cam2img.shape[0], 1), dtype=np.float32)], axis=1)
        
        
        for i in range(gt_bboxes_3d.shape[0]):
            gt_box = gt_bboxes_3d[i]
            loc = gt_box[:3]
            lhw = gt_box[3:6]
            rotation_y = gt_box[6]
            dim = lhw[[1,2,0]]
            # loc[1] = loc[1] - lhw[1] / 2 

            box_3d = compute_box_3d_camera(dim, loc, rotation_y)
            box_2d = project_to_image(box_3d, P)
            image = draw_box_3d(img, box_2d, c=c)
        cv2.imwrite("demo.jpg", image)


    def __call__(self, results):
        print(results.keys())
        self.visualize(results['img'], results["gt_bboxes_3d"], results["cam2img"])
        results['img_pair'] = list()
        results["lidar2cam_pair"], results["cam_intrinsic_pair"], results["lidar2img_pair"] = list(), list(), list()
        for idx, image in enumerate(results['img']):
            lidar2cam = results["lidar2cam"][idx].copy()
            lidar2img = results["lidar2img"][idx].copy()            
            cam_intrinsic = results["cam_intrinsic"][idx].copy()
            image = results["img"][idx].copy()
    
            cam_intrinsic_rectify, image = self.reactify_ratio_params(image, cam_intrinsic)
            lidar2cam_rectify, lidar2img_rectify, image = self.reactify_roll_params(image, lidar2cam, cam_intrinsic)
            lidar2cam_rectify, lidar2img_rectify, image = self.reactify_pitch_params(image, lidar2cam_rectify, cam_intrinsic)                            
            
            if results["cache_sample"]:
                results["lidar2cam_pair"].append(lidar2cam_rectify)
                results["cam_intrinsic_pair"].append(cam_intrinsic_rectify)
                results["lidar2img_pair"].append(lidar2img_rectify)
                results["img_pair"].append(image)
            else:
                results["lidar2cam_pair"].append(lidar2cam)
                results["cam_intrinsic_pair"].append(cam_intrinsic)
                results["lidar2img_pair"].append(lidar2img)
                results["img_pair"].append(results["img"][idx])

                results["lidar2cam"][idx] = lidar2cam_rectify
                results["cam_intrinsic"][idx] = cam_intrinsic_rectify
                results["lidar2img"][idx] = lidar2img_rectify
                results["img"][idx] = image
        return results
    
    def get_M(self, R, K, R_r, K_r):
        R_inv = np.linalg.inv(R)
        K_inv = np.linalg.inv(K)
        M = np.matmul(K_r, R_r)
        M = np.matmul(M, R_inv)
        M = np.matmul(M, K_inv)
        return M
    