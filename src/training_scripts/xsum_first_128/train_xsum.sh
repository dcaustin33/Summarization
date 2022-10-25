export PYTHONPATH='../';
python3 train_xsum.py \
            --model_name google/pegasus-large \
            --batch_size 6 \
            --max_length 128 \
            --steps 20000 \
            --name Pegasus_XSum_first_128 \
            --log_n_train_steps 100 \
            --log_n_val_steps 10 \
            --checkpoint_every_n_steps 1000 \
            --warmup_steps 100 \
            --val_step 2 \
            --workers 4 \
            -log;
#sudo shutdown -h;
            #-checkpoint \
            #--checkpoint_path checkpoints/Pegasus_XSum_first_128/Pegasus_XSum_first_128_checkpoint.pt \