#!/usr/bin/env bash

# BTTA-enhanced launcher (adapted from gap/run.sh)
# Usage:
#   bash run_btta.sh [model_name]

cd /g/data/hn98/du/exlts/ddd2

model_name=${1:-DLinear}

# BTTA core switches
btta_stage=${BTTA_STAGE:-full}            # pretrain | bta | btta | full
lambda_s=${LAMBDA_S:-0.2}
lambda_d=${LAMBDA_D:-0.2}
lambda_m=${LAMBDA_M:-0.2}
mu_grad=${MU_GRAD:-0.0}
stable_agg_scale=${STABLE_AGG_SCALE:-12}
trend_kernel=${TREND_KERNEL:-25}
measurement_mask_ratio=${MEASUREMENT_MASK_RATIO:-0.15}

for pred_len in 96 192 336 720; do
  python -u run.py \
    --train_seed 2024 \
    --gap_day 548 \
    --samle_rate 0.1 \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../../data/ \
    --data_path pems10_all_common_flow.csv \
    --model_id pems10_all_96_${pred_len}_btta \
    --model ${model_name} \
    --data custom \
    --target '' \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 107 \
    --dec_in 107 \
    --c_out 107 \
    --des 'BTTA' \
    --itr 1 \
    --learning_rate 0.0005 \
    --train_epochs 200 \
    --patience 5 \
    --lradj 'type3' \
    --enable_btta \
    --btta_stage ${btta_stage} \
    --lambda_s ${lambda_s} \
    --lambda_d ${lambda_d} \
    --lambda_m ${lambda_m} \
    --mu_grad ${mu_grad} \
    --stable_agg_scale ${stable_agg_scale} \
    --trend_kernel ${trend_kernel} \
    --measurement_mask_ratio ${measurement_mask_ratio} \
    >> ${model_name,,}_pems10_gap15_in96_out${pred_len}_btta.log 2>&1
done
