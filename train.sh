for i in 2
do
    python main.py --model BiO --train_mode 1 --in_ch 32 --patience 200 --DID 3 --lr 0.0001 --fold_i $i --batch_size 4 --lr_schedule 3 --nfold 3 \
    --sym --stn --weight 1. 10. 0. 5. 0.1 0 --data_root ../ventricle/data/data --max_epoch 400
done
