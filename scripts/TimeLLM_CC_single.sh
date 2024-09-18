
comment='CongestionControl'
python CC.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/CC/ \
  --data_path output.csv \
  --model_id CC_512_96 \
  --model  TimeLLM\
  --data CC  \
  --features M \
  --seq_len 96 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 1 \
  --learning_rate 0.01 \
  --llm_layers 32 \
  --train_epochs 10 \
  --model_comment 'CongestionControl'
