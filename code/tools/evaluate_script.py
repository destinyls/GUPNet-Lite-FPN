import os
import sys
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import shutil

import warnings
warnings.filterwarnings("ignore")

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.tester_helper import Tester
from tools.train_val import create_logger

parser = argparse.ArgumentParser(description='implementation of GUPNet')
parser.add_argument('--config', type=str, default = 'experiments/config.yaml')
args = parser.parse_args()

def main():  
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'],exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'],'test.log'))    

    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])
    # build model
    model = build_model(cfg['model'],train_loader.dataset.cls_mean_size)

    output_dir = cfg['tester']['output_dir']
    best_mAP = 0
    for ckpt in os.listdir(cfg['tester']['output_dir']):
        if ".pth" not in ckpt:
            continue
        iteration = int(ckpt.split('_')[2].split('.')[0])
        ckpt = os.path.join(output_dir, ckpt)
        model_state = torch.load(ckpt)["model_state"]
        model.load_state_dict(model_state)
        tester = Tester(cfg['tester'], model, val_loader, logger)
        mAP_3d_moderate = tester.test(iteration)
        if mAP_3d_moderate > best_mAP:
            best_mAP = mAP_3d_moderate
            shutil.copyfile(ckpt, os.path.join(output_dir, "best_checkpoints.pth"))

if __name__ == '__main__':
    main()