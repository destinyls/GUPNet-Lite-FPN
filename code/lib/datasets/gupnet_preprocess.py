import cv2
import math

import numpy as np
from mmdet.datasets.builder import PIPELINES
from lib.datasets.visual_utils import compute_box_3d_camera, project_to_image, draw_box_3d
from lib.datasets.utils import angle2class

def ry2alpha(ry, u, cam2img):
    cu = cam2img[0, 2]
    fu = cam2img[0, 0]
    alpha = ry - np.arctan2(u - cu, fu)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha

def alpha2ry(alpha, u, cam2img):
    cu = cam2img[0, 2]
    fu = cam2img[0, 0]
    ry = alpha + np.arctan2(u - cu, fu)
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def rect_to_img(pts_rect, cam2img):
    pts_2d = np.matmul(cam2img, pts_rect)[:,0]
    pts_rect_depth = pts_2d[2]
    pts_img = pts_2d[:2] / pts_2d[2]
    return pts_img, pts_rect_depth

def img_to_rect(u, v, depth_rect, cam2img):
    cu = cam2img[0, 2]
    cv = cam2img[1, 2]
    fu = cam2img[0, 0]
    fv = cam2img[1, 1]
    tx = cam2img[0, 3] / (-1 * fu)
    ty = cam2img[1, 3] / (-1 * fv)
    x = ((u - cu) * depth_rect) / fu + tx
    y = ((v - cv) * depth_rect) / fv + ty
    pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    return pts_rect

@PIPELINES.register_module()
class GUPNetPreprocess(object):
    def __init__(self, class_names, split="train"):
        self.downsample = 4
        self.resolution = np.array([1600, 928])  # W * H
        self.split = split
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.writelist = class_names
        self.max_objs = 80
        self.use_3d_center = True

        self.cls2id = dict()
        for i in range(len(class_names)):
            self.cls2id[class_names[i]] = i

        # l, w, h
        self.cls_mean_size = np.array(
                    [[2.114256, 1.620300, 0.927272],
                    [0.791118, 1.279516, 0.718182],
                    [0.923508, 1.867419, 0.845495],
                    [0.591958, 0.552978, 0.827272],
                    [0.699104, 0.454178, 0.75625],
                    [0.69519, 1.346299, 0.736364],
                    [0.528526, 1.002642, 1.172878],
                    [0.500618, 0.632163, 0.683424],
                    [0.404671, 1.071108, 1.688889],
                    [0.76584, 1.398258, 0.472728]])

    def visualize(self, img, gt_bboxes_3d, cam2img, c=(0, 255, 0)):
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = img.permute(1,2,0).numpy().copy()
        img = img * std + mean
        P = np.concatenate(
            [cam2img, np.zeros((cam2img.shape[0], 1), dtype=np.float32)], axis=1)
        
        for i in range(gt_bboxes_3d.shape[0]):
            gt_box = gt_bboxes_3d[i]
            loc = gt_box[:3]
            lhw = gt_box[3:6]
            rotation_y = gt_box[6]
            dim = lhw[[1,2,0]]
            box_3d = compute_box_3d_camera(dim, loc, rotation_y)
            box_2d = project_to_image(box_3d, P)
            img = draw_box_3d(img, box_2d, c=c)
        cv2.imwrite("demo.jpg", img)

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
    
    def gaussian_radius(self, bbox_size, min_overlap=0.7):
        height, width = bbox_size
        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)
    
    def get_alpha(self, ry, z, x):
        alpha = ry - (math.pi * 0.5 - math.atan2(z, x))
        if alpha > math.pi:
            alpha -= 2 * math.pi
        if alpha <= -math.pi:
            alpha += 2 * math.pi
        return alpha

    def __call__(self, results):
        features_size = self.resolution // self.downsample  # W * H
        img = results["img"].data
        cam2img = np.array(results["cam2img"])
        img_size = np.array([img.shape[2], img.shape[1]])
        center = img_size / 2
        coord_range = np.array([center - img_size / 2, center + img_size / 2]).astype(np.float32)                   
    
        if self.split == 'train':
            gt_bboxes_3d = results["gt_bboxes_3d"].data.tensor.numpy()
            gt_labels_3d = results["gt_labels_3d"].data.numpy()
            gt_bboxes = results["gt_bboxes"].data.numpy()
            attr_labels = results["attr_labels"].data.numpy()
            # self.visualize(img, gt_bboxes_3d, cam2img)

            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            attrs = np.zeros((self.max_objs), dtype=np.uint8)
            velocity = np.zeros((self.max_objs, 2), dtype=np.float32)

            object_num = gt_bboxes_3d.shape[0] if gt_bboxes_3d.shape[0] < self.max_objs else self.max_objs
            for i in range(object_num):
                # filter objects by writelist
                if self.class_names[gt_labels_3d[i]] not in self.writelist:
                    continue
                gt_bbox3d = gt_bboxes_3d[i]
                cls_id = gt_labels_3d[i]
                loc = gt_bbox3d[:3]
                lhw = gt_bbox3d[3:6]
                r_y = gt_bbox3d[6]
                velo = gt_bbox3d[7:]
                # filter inappropriate samples by difficulty
                if loc[-1] < 2: continue
                # process 2d bbox & get 2d center
                bbox_2d = gt_bboxes[i].copy()
                bbox_2d[:] /= self.downsample

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = loc + [0, -lhw[1] / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(3, -1)  # shape adjustment (3, N)
                center_3d, _ = rect_to_img(center_3d, cam2img)
                center_3d /= self.downsample      
            
                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = self.gaussian_radius((w, h))
                radius = max(0, int(radius))
                cls_ids[i] = cls_id
                self.draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h

                # encoding depth
                depth[i] = loc[-1]
                # encoding heading angle
                heading_angle = ry2alpha(r_y, (gt_bboxes[i][0] + gt_bboxes[i][2])/2, cam2img)                
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([lhw[1], lhw[2], lhw[0]], dtype=np.float32)
                mean_size = self.cls_mean_size[cls_id]
                size_3d[i] = src_size_3d[i] - mean_size
                mask_2d[i] = 1
                mask_3d[i] = 1
                velocity[i] = velo
                attrs[i] = attr_labels[i]
                            
            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d,
                       'attrs' : attrs,
                       'velocity': velocity}
        else:
            targets = {}
        # collect return data
        # print("results: ", results["img_info"].keys())
        info = {'img_id': results["img_info"]["file_name"],
                'token': results["img_info"]["token"],
                # 'img_info': results["img_info"],
                'img_size': img_size,
                'coord_range': coord_range,                
                'bbox_downsample_ratio': img_size / features_size}   
        results["info"] = info
        results["targets"] = targets
        return results
    