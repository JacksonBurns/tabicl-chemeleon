
wget --no-clobber https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt
wget --no-clobber https://huggingface.co/jingang/TabICL/resolve/main/tabicl-regressor-v2-20260212.ckpt

torchrun --standalone --nproc_per_node=1 -m tabicl.train \
    --device cuda \
    --dtype float32 \
    --max_steps 10000 \
    --batch_size 64 \
    --micro_batch_size 1 \
    --lr 1e-5 \
    --muon True \
    --beta1 0.9 \
    --weight_decay 0.01 \
    --scheduler cosine_with_restarts \
    --warmup_proportion 0.01 \
    --cosine_num_cycles 1 \
    --cosine_amplitude_decay 1 \
    --cosine_lr_end 1e-7 \
    --gradient_clipping 1.0 \
    --regression_method quantile \
    --num_quantiles 999 \
    --prior_type chemeleon \
    --prior_device cpu \
    --n_jobs 16 \
    --batch_size_per_gp 1 \
    --min_features 2048 \
    --max_features 2048 \
    --min_seq_len 400 \
    --max_seq_len 60000 \
    --log_seq_len True \
    --seq_len_per_gp False \
    --min_train_size 0.79 \
    --max_train_size 0.81 \
    --checkpoint_path tabicl-regressor-v2-20260212.ckpt \
    --only_load_model True \
    --checkpoint_dir ./checkpoints
