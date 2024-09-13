model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

master_port=15294
num_process=2
batch_size=8
d_model=32
d_ff=128

comment='CongestionControl'
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/CC/ \
  --data_path output.csv \
  --model_id CC_512_96 \
  --model $model_name \
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
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
