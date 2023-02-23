export PYTHONPATH=./$PYTHONPATH
export TORCH_HOME=./$TORCH_HOME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train_val.py --config experiments/config.yaml
CUDA_VISIBLE_DEVICES=3 python tools/evaluate_script.py --config experiments/config.yaml
