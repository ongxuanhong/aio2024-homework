set -x

deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/Llama-3.2-1B-rm-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 96 \
   --micro_train_batch_size 8 \
   --pretrain thuanan/Llama-3.2-1B-Instruct-Chat-sft \
   --value_head_prefix score \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 2 \
   --learning_rate 5e-6 \
   --dataset thuanan/Vi-Alpaca-Preference \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --adam_offload \
   --lora_rank 16 \
   --lora_alpha 32