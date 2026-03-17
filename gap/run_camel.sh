#!/usr/bin/env bash

# CAMEL-enhanced launcher
# Usage: bash run_camel.sh [model_name]

cd /g/data/hn98/du/exlts/ddd2

model_name=${1:-CAMEL}
gap_years=${CAMEL_GAP_YEARS:-1.5}

for pred_len in 96 192 336 720; do
  python -u run.py \
    --train_seed 2024 \
    --gap_day 548 \
    --samle_rate 0.1 \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../../data/ \
    --data_path pems10_all_common_flow.csv \
    --model_id pems10_all_96_${pred_len}_camel \
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
    --des 'CAMEL' \
    --itr 1 \
    --learning_rate 0.0005 \
    --train_epochs 200 \
    --patience 5 \
    --lradj 'type3' \
    --camel_gap_years ${gap_years} \
    --lambda_mem 0.10 \
    --lambda_ode 0.05 \
    --lambda_smooth 0.01 \
    >> ${model_name,,}_pems10_gap15_in96_out${pred_len}_camel.log 2>&1
done
