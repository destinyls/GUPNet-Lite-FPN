import os
import cv2
import tqdm
import time

import torch
import numpy as np

from lib.datasets.visual_utils import compute_box_3d_camera, project_to_image, draw_box_3d
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from mmdet3d.core.bbox import CameraInstance3DBoxes, get_box_type

class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.bbox_code_size = 9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)

    def visualize(self, img, dets, cam2img, c=(0, 255, 0)):
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = img.permute(1,2,0).cpu().numpy().copy()
        img = img * std + mean

        for i in range(len(dets)):
            det = dets[i]
            cls_id = det[0]
            loc = det[9:12]
            dim = det[6:9]
            rotation_y = det[12]

            box_3d = compute_box_3d_camera(dim, loc, rotation_y)
            box_2d = project_to_image(box_3d, cam2img)
            img = draw_box_3d(img, box_2d, c=c)
        cv2.imwrite("demo_123.jpg", img)
        time.sleep(3)

    def format_results(self, bboxes_list):
        results = list()
        for i in range(len(bboxes_list)):
            bboxes = bboxes_list[i]
            res = dict()
            if len(bboxes) > 0:
                gt_bboxes_cam3d, gt_labels_cam3d, gt_scores_cam3d, gt_attrs_3d = list(), list(), list(), list()
                for bbox in bboxes:
                    cls_id = bbox[0]
                    hwl = bbox[6:9]
                    xyz = bbox[9:12]
                    xyz[1] = xyz[1] - hwl[0] / 2
                    ry = bbox[12]
                    score = bbox[13]
                    velocity = bbox[14:16]
                    attrs = int(bbox[16])
                    gt_bboxes_cam3d.append([xyz[0], xyz[1], xyz[2], hwl[2], hwl[0], hwl[1], ry, velocity[0], velocity[1]])
                    gt_labels_cam3d.append(cls_id)
                    gt_scores_cam3d.append(score)
                    gt_attrs_3d.append(attrs)
                gt_bboxes_cam3d = np.array(gt_bboxes_cam3d)
                gt_scores_cam3d = np.array(gt_scores_cam3d)
                gt_labels_cam3d = np.array(gt_labels_cam3d)
                gt_attrs_3d = np.array(gt_attrs_3d)
            else:
                gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size), dtype=np.float32)
                gt_labels_cam3d = np.zeros((0, 1), dtype=np.float32)
                gt_scores_cam3d = np.zeros((0, 1), dtype=np.float32)
                gt_attrs_3d = np.zeros((0, 1), dtype=np.float32)

            gt_bboxes_cam3d = torch.tensor(gt_bboxes_cam3d)
            gt_labels_cam3d = torch.tensor(gt_labels_cam3d)
            gt_scores_cam3d = torch.tensor(gt_scores_cam3d)
            gt_attrs_3d = torch.tensor(gt_attrs_3d)
            gt_bboxes_cam3d = CameraInstance3DBoxes(
                gt_bboxes_cam3d,
                box_dim=gt_bboxes_cam3d.shape[-1],
                origin=(0.5, 1.0, 0.5))  
            res['boxes_3d'] = gt_bboxes_cam3d
            res['scores_3d'] = gt_scores_cam3d
            res['labels_3d'] = gt_labels_cam3d
            res['attrs_3d'] = gt_attrs_3d
            results.append({"img_bbox": res})
        return results

    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        bboxes_list = list()
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            # the outputs of centernet
            outputs = self.model(inputs,coord_ranges, calibs,K=50,mode='test')
            dets = extract_dets_from_outputs(outputs=outputs, K=50)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            # calibs = [self.data_loader.dataset.get_calib(index)  for index in info['img_id']]
            calibs = calibs.detach().cpu().numpy() 
            info = {key: val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val  for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets, bboxes = decode_detections(dets = dets,
                                     info = info,
                                     calibs = calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold = self.cfg['threshold'])
            
            # self.visualize(inputs[0], list(dets.values())[0], calibs[0])
            bboxes_list.append(bboxes)
            results.update(dets)
            progress_bar.update()
        # save the result for evaluation.
        # self.save_results(results)

        results = self.format_results(bboxes_list)
        self.data_loader.dataset.evaluate(results)
        progress_bar.close()

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, img_id + '.txt' if isinstance(img_id, str) else '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            if len(results[img_id]) == 0:
                f.write('\n')
                f.write('\n')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()
            self.check_last_line_break(out_path)

    def check_last_line_break(self, out_path):
        f = open(out_path, 'rb+')
        try:
            f.seek(-1, os.SEEK_END)
        except:
            pass
        else:
            if f.__next__() == b'\n':
                f.seek(-1, os.SEEK_END)
                f.truncate()
        f.close()
