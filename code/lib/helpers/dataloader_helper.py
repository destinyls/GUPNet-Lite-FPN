import numpy as np

from mmcv import Config
from mmdet3d.datasets.builder import build_dataset

from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_dataloader_custom(cfg):
    # --------------  build kitti dataset -----------------
    if cfg['type'] == 'kitti':
        train_set = KITTI(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=2,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_set = KITTI(root_dir=cfg['root_dir'], split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=2,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        test_set = KITTI(root_dir=cfg['root_dir'], split='test', cfg=cfg) 
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=2,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader
    elif cfg['type'] == 'nuscenes-mmdet3d':
        cfg = Config.fromfile("lib/datasets/nus-mono3d.py")
        train_dataset = build_dataset(cfg.data.train)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=6,
                                  num_workers=8,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        
        val_dataset = build_dataset(cfg.data.val)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                num_workers=8,
                                worker_init_fn=my_worker_init_fn,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
        return train_loader, val_loader, None
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

