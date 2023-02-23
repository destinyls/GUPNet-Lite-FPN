export PYTHONPATH=./$PYTHONPATH
export TORCH_HOME=./$TORCH_HOME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python tools/train_val.py --config experiments/config.yaml
