CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nporc_per_node 4 main.py train --ep 200 --model 0 --ddp
