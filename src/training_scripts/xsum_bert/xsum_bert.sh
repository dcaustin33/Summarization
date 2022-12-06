export PYTHONPATH='../';

python3 train_xsum_bert.py \
            --model_name google/pegasus-large \
            --batch_size 4 \
            --max_length 512 \
            --steps 4000 \
            --name Pegasus_XSum_BERT_512 \
            --log_n_train_steps 100 \
            --log_n_val_steps 400 \
            --checkpoint_every_n_steps 200 \
            --warmup_steps 100 \
            --val_step 20 \
            --workers 4 \
            --num_beams 4 \
            -log;

#python3 generate_and_evaluate.py \
#            --model_name google/pegasus-large \
#            --batch_size 2 \
#            --max_length 512 \
#            --name Eval_Pegasus_XSum_2_Power_law_512   \
#            --log_n_val_steps 200 \
#            --val_steps 200 \
#            --write_steps 10 \
#            --num_beams 4 \
#            --divisor 2 \
#            --model_path checkpoints/Pegasus_XSum_2_Power_law_512/Pegasus_XSum_2_Power_law_512_Final.pt \
#            --workers 4 \
#            -log;