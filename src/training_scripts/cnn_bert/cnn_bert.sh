export PYTHONPATH='../';

python3 train_cnn_bert.py \
            --model_name google/pegasus-large \
            --batch_size 4 \
            --max_length 512 \
            --steps 4000 \
            --name Pegasus_CNN_BERT_512 \
            --log_n_train_steps 100 \
            --log_n_val_steps 400 \
            --checkpoint_every_n_steps 200 \
            --warmup_steps 100 \
            --val_step 20 \
            --workers 4 \
            --num_beams 4 \
            -log;

python3 train_cnn_bert.py \
            --model_name google/pegasus-large \
            --batch_size 4 \
            --max_length 256 \
            --steps 4000 \
            --name Pegasus_CNN_BERT_256 \
            --log_n_train_steps 100 \
            --log_n_val_steps 400 \
            --checkpoint_every_n_steps 200 \
            --warmup_steps 100 \
            --val_step 20 \
            --workers 4 \
            --num_beams 4 \
            -log;

python3 train_cnn_bert.py \
            --model_name google/pegasus-large \
            --batch_size 4 \
            --max_length 128 \
            --steps 4000 \
            --name Pegasus_CNN_BERT_128 \
            --log_n_train_steps 100 \
            --log_n_val_steps 400 \
            --checkpoint_every_n_steps 200 \
            --warmup_steps 100 \
            --val_step 20 \
            --workers 4 \
            --num_beams 4 \
            -log;

sudo shutdown -h;