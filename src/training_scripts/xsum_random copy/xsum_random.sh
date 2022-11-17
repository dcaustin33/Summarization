export PYTHONPATH='../';

#python3 train_xsum_random.py \
#            --model_name google/pegasus-large \
#            --batch_size 4 \
#            --max_length 512 \
#            --steps 4000 \
#            --name Pegasus_xsum_random_512 \
#            --log_n_train_steps 100 \
#            --log_n_val_steps 400 \
#            --checkpoint_every_n_steps 1000 \
#            --warmup_steps 100 \
#            --val_step 20 \
#            --workers 4 \
#            --num_beams 4 \
#            -log;

python3 generate_and_evaluate.py \
            --model_name google/pegasus-large \
            --batch_size 2 \
            --max_length 512 \
            --name Eval_Pegasus_xsum_random_512 \
            --log_n_val_steps 100 \
            --val_steps 100 \
            --write_steps 10 \
            --num_beams 4 \
            --model_path checkpoints/Pegasus_xsum_random_512/Pegasus_xsum_random_512_Final.pt \
            --workers 4 \
            -log;


#python3 train_xsum_random.py \
#            --model_name google/pegasus-large \
#            --batch_size 4 \
#            --max_length 256 \
#            --steps 4000 \
#            --name Pegasus_xsum_random_256 \
#            --log_n_train_steps 100 \
#            --log_n_val_steps 400 \
#            --checkpoint_every_n_steps 1000 \
#            --warmup_steps 100 \
#            --val_step 20 \
#            --workers 4 \
#            --num_beams 4 \
#            -log;

python3 generate_and_evaluate.py \
            --model_name google/pegasus-large \
            --batch_size 2 \
            --max_length 256 \
            --name Eval_Pegasus_xsum_random_256 \
            --log_n_val_steps 100 \
            --val_steps 100 \
            --write_steps 10 \
            --num_beams 4 \
            --model_path checkpoints/Pegasus_xsum_random_256/Pegasus_xsum_random_256_Final.pt \
            --workers 4 \
            -log;

#python3 train_xsum_random.py \
#            --model_name google/pegasus-large \
#            --batch_size 4 \
#            --max_length 128 \
#            --steps 4000 \
#            --name Pegasus_xsum_random_128 \
#            --log_n_train_steps 100 \
#            --log_n_val_steps 400 \
#            --checkpoint_every_n_steps 1000 \
#            --warmup_steps 100 \
#            --val_step 20 \
#            --workers 4 \
#            --num_beams 4 \
#            -log;

python3 generate_and_evaluate.py \
            --model_name google/pegasus-large \
            --batch_size 2 \
            --max_length 128 \
            --name Eval_Pegasus_xsum_random_128 \
            --log_n_val_steps 100 \
            --val_steps 100 \
            --write_steps 10 \
            --num_beams 4 \
            --model_path checkpoints/Pegasus_xsum_random_128/Pegasus_xsum_random_128_Final.pt \
            --workers 4 \
            -log;

#sudo shutdown -h;