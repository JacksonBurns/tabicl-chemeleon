
wget --no-clobber https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt
wget --no-clobber https://huggingface.co/jingang/TabICL/resolve/main/tabicl-regressor-v2-20260212.ckpt

python -m tabicl.train \
    --prior_dir data_chemeleon \
    --device cuda \
    --dtype float16 \
    --max_steps 1024 \
    --batch_size 8 \
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
    --norm_type layernorm_nobias \
    --col_feature_group True \
    --col_target_aware True \
    --col_affine False \
    --col_ssmax True \
    --icl_ssmax True \
    --ssmax_type qassmax-mlp-elementwise \
    --col_nhead 8 \
    --icl_nhead 8 \
    --row_rope_interleaved False \
    --zero_init False \
    --prior_type chemeleon \
    --prior_device cuda \
    --n_jobs 0 \
    --min_features 2048 \
    --max_features 2048 \
    --min_seq_len 128 \
    --max_seq_len 2048 \
    --log_seq_len True \
    --seq_len_per_gp False \
    --min_train_size 0.79 \
    --max_train_size 0.81 \
    --checkpoint_path tabicl-regressor-v2-20260212.ckpt \
    --only_load_model True \
    --checkpoint_dir ./checkpoints
    