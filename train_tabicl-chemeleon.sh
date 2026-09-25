
wget --no-clobber https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt
wget --no-clobber https://huggingface.co/jingang/TabICL/resolve/main/tabicl-regressor-v2-20260212.ckpt

# TODO: train from scratch with this same setup? can maybe turn off the column variability approximations?
python -m tabicl.train \
    --prior_dir /media/jackson/pciedisk/MiniCheMeleon/MiniCheMeleon-TabICL/cached_minichemeleon_embeddings \
    --device cuda \
    --dtype float16 \
    --max_steps 150 \
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
    --checkpoint_path tabicl-regressor-v2-20260212.ckpt \
    --only_load_model True \
    --checkpoint_dir /media/jackson/pciedisk/MiniCheMeleon/MiniCheMeleon-TabICL/incremental_pretrain_checkpoints
