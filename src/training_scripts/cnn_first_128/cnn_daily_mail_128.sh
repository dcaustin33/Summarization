export PYTHONPATH='../';
python3 train_cnn_daily_mail.py \
            --model_name google/pegasus-large \
            --batch_size 6 \
            --max_length 128 \
            --steps 4000 \
            --name CNN_Daily_Mail_128 \
            --log_n_train_steps 100 \
            --log_n_val_steps 200 \
            --checkpoint_every_n_steps 1000 \
            --warmup_steps 100 \
            --val_step 2 \
            --workers 4 \
            -log;

#python3 generate_and_evaluate.py \
#            --model_name google/pegasus-large \
#            --batch_size 6 \
#            --max_length 128 \
#            --name Eval_Pegasus_XSum_first_128 \
#            --log_n_val_steps 10 \
#            --val_steps 10 \
#            --model_path checkpoints/Pegasus_XSum_first_128/Pegasus_XSum_first_128_Final.pt \
#            --workers 4 ;#\
#            #-log;
#sudo shutdown -h;