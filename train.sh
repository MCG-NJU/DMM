accelerate launch \
    train.py \
    --project_dir xxx \
    --models_json_file ./configs/example.json \
    --mixed_precision fp16 \
    --pretrained_model_path path/to/stable-diffusion-v1-5 \
    --vae_model_path path/to/vae \
    --num_train_steps 200000 \
    --gradient_accumulation_steps 1 \
    --train_batch_size 50 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --use_feat_loss \
    --feat_loss_type l2 \
    --feat_loss_weight 0.002 \
    --log_interval 50 \
    --save_interval 1000 \
