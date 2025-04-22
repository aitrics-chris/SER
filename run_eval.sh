unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 PYTHONHASHSEED=1 python lincls_parallel.py --epochs 30 --val_epoch 27 --config eval1.yaml --seed 1