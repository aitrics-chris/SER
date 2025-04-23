# DINO
# alias python='/home/andy/anaconda3/envs/p260/bin/python3.9'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 PYTHONHASHSEED=1 /home/andy/anaconda3/envs/p260/bin/python3.9 -m torch.distributed.launch --nproc_per_node=8 --master_port=23456 main_dino.py --lr 0.0001 --num_workers 4 --arch vit_small --epochs 100 --warmup_epochs 10 \
    --batch_size_per_gpu 256 --local_crops_number 0 --norm_last_layer False --momentum_teacher 0.982 --equiv-scale 0.7 1.3 --equiv-lambda 0.3 --equiv-layer 3 --warmup-epochs-scheduler 0 \
    --rest-epochs-scheduler 0 --equiv-ratio-start 0.01 --clip_grad 0.3 --equiv-ratio-end 0.0 --equiv-mode erl --temperature 0.4 --optimizer adamw --tag ep100_adamw_mt981 \
    --data_path /home/andy/datasets/imagenet/train/ --output_dir ./results --ckpt_dir ./ckpt
    # > out/DINO_equiv_vit_small_0.0001_contrastive_0.7_1.3_0.75_0.3_fix_3_0_0_0.01_0.0_clipgrad_0.3_0.4_phastos_ep100_adamw_mt981.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 PYTHONHASHSEED=1 /home/andy/anaconda3/envs/p260/bin/python3.9 -m torch.distributed.launch --nproc_per_node=8 --master_port=23456 main_dino.py --lr 0.0001 --num_workers 4 --arch vit_small --epochs 100 --warmup_epochs 10 \
#     --batch_size_per_gpu 16 --local_crops_number 0 --norm_last_layer False --momentum_teacher 0.982 --equiv-scale 0.7 1.3 --equiv-lambda 0.3 --equiv-layer 12 --warmup-epochs-scheduler 0 \
#     --rest-epochs-scheduler 0 --equiv-ratio-start 0.01 --clip_grad 0.3 --equiv-ratio-end 0.0 --equiv-mode stl --temperature 0.4 --optimizer adamw --tag ep100_adamw_mt981 \
#     --stl-trans-backbone 128-128 --stl-trans-projector 128-128 --stl-projector 512-128 --stl-inv-weight 1.0 --stl-equi-weight 1.0 --stl-trans-weight 0.1 \
#     --data_path /home/andy/datasets/imagenet/train/ --output_dir ./results --ckpt_dir ./ckpt
#     # \ > out/DINO_equiv_vit_small_0.0001_contrastive_0.7_1.3_0.75_0.3_fix_3_0_0_0.01_0.0_clipgrad_0.3_0.4_phastos_ep100_adamw_mt981.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 PYTHONHASHSEED=1 /home/andy/anaconda3/envs/p260/bin/python3.9 -m torch.distributed.launch --nproc_per_node=8 --master_port=23456 main_dino.py --lr 0.0001 --num_workers 4 --arch vit_small --epochs 100 --warmup_epochs 10 \
#     --batch_size_per_gpu 16 --local_crops_number 0 --norm_last_layer False --momentum_teacher 0.982 --equiv-scale 0.7 1.3 --equiv-lambda 0.3 --equiv-layer 12 --warmup-epochs-scheduler 0 \
#     --rest-epochs-scheduler 0 --equiv-ratio-start 0.01 --clip_grad 0.3 --equiv-ratio-end 0.0 --equiv-mode equimod --temperature 0.4 --optimizer adamw --tag ep100_adamw_mt981 \
#     --stl-trans-backbone 128-128 --stl-trans-projector 128-128 --stl-projector 512-128 --stl-inv-weight 1.0 --stl-equi-weight 1.0 --stl-trans-weight 0.1 \
#     --data_path /home/andy/datasets/imagenet/train/ --output_dir ./results --ckpt_dir ./ckpt
#     # \ > out/DINO_equiv_vit_small_0.0001_contrastive_0.7_1.3_0.75_0.3_fix_3_0_0_0.01_0.0_clipgrad_0.3_0.4_phastos_ep100_adamw_mt981.txt

# MoCo
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 PYTHONHASHSEED=1 python -m torch.distributed.launch --nproc_per_node=8 mimic_batch.py --lr 0.0003 --batch-size 2048 --epochs 100 --warmup-epochs 10 \
#     --optimizer adamw --weight-decay 0.1 --moco-m-cos --moco-t 0.2 --equiv-scale 0.7 1.3 --equiv-lambda 1.0 --equiv-layer 3 --warmup-epochs-scheduler 0 --rest-epochs-scheduler 0 \
#     --equiv-ratio-start 0.01 --clip_grad 0.0 --equiv-ratio-end 0.0 --equiv-mode erl --temperature-equiv 0.5 --tag _ > out/MoCo_equiv_vit_small_0.0024_0.7_1.3_0.75_1.0_fix_3_0_0_0.01_0.0_clipgrad_0.0_0.5_sersi_ep100__.txt

# Barlowtwins
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 PYTHONHASHSEED=1 python -m torch.distributed.launch --nproc_per_node=8 mimic_barlowtwins.py --arch vit_small --learning-rate-weights 0.0001 --learning-rate-biases 0.0048 \
#     --batch-size 2048 --epochs 100 --equiv-scale 0.7 1.3 --equiv-lambda 1.0 --equiv-layer 3 --warmup-epochs-scheduler 0 --clip_grad 0.3 --equiv-mode erl --temperature-equiv 0.4 \
#     --rest-epochs-scheduler 0 --equiv-ratio-start 0.01 --equiv-ratio-end 0.0 --tag unfreeze > out/BT_equiv_vit_small_0.0001_0.0048_0.7_1.3_0.75_1.0_fix_3_0_0_0.01_0.0_clipgrad_0.3_0.4_phastos_ep100_unfreeze.txt
