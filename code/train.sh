CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config experiments/config.yaml
CUDA_VISIBLE_DEVICES=0 python tools/eval.py --config experiments/config.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config experiments/config_002.yaml
CUDA_VISIBLE_DEVICES=0 python tools/eval.py --config experiments/config_002.yaml