import os
import tqdm
import time

import json
import torch
import numpy as np

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from lib.evaluation.kitti_utils.eval import kitti_eval
from lib.evaluation.kitti_utils import kitti_common as kitti

class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            # the outputs of centernet
            outputs = self.model(inputs,coord_ranges,calibs,K=self.data_loader.dataset.max_objs, mode='test')
            dets = extract_dets_from_outputs(outputs=outputs, K=self.data_loader.dataset.max_objs)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.data_loader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets = decode_detections(dets = dets,
                                     info = info,
                                     calibs = calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold = self.cfg['threshold'])
            results.update(dets)
            progress_bar.update()

        progress_bar.close()
        
        # with open(os.path.join(self.cfg['output_dir'], "results.json"), 'w') as f:
        #    json.dump(results, f)
        # save the result for evaluation.
        evaluation_path = self.cfg['output_dir']
        pred_label_path = os.path.join(self.cfg['output_dir'], 'data')
        gt_label_path = os.path.join(self.data_loader.dataset.root_dir, "KITTI/training/label_2/")
        self.save_results(results, output_dir=pred_label_path)        
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"])
        print(result)
        if ret_dict is not None:
            mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
            with open(os.path.join(checkpoints_path, 'epoch_result_{:07d}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
                f.write(result)
        print(result)

    def save_results(self, results, output_dir='./outputs'):
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
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
        
if __name__ == "__main__":
    gt_label_path = "/root/Dataset/kitti_dataset/training/label_2"
    gt_label_path = "/workspace/tsing-adept/datasets/kitti/training/label_2"
    pred_annos, image_ids = kitti.get_label_annos(gt_label_path, return_ids=True)
    gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
    result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"])
    print(result)
    if ret_dict is not None:
        mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
        val_mAP.append(mAP_3d_moderate)
        with open(os.path.join(checkpoints_path, "val_mAP.json"),'w') as file_object:
            json.dump(val_mAP, file_object)
        with open(os.path.join(checkpoints_path, 'epoch_result_{:07d}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
            f.write(result)
    print(result)







