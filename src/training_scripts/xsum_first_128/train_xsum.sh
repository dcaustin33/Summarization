export PYTHONPATH='../';
python3 train_xsum.py \
            --model_name google/pegasus-large \
            --batch_size 4 \
            --max_length 128 \
            --steps 20000 \
            --name Pegasus_XSum_first_128 \
            --log_n_train_steps 100 \
            --log_n_val_steps 1000 \
            --checkpoint_every_n_steps 1000 \
            --warmup_steps 100 \
            --val_step 5 \
            --workers 4 \
            -log;