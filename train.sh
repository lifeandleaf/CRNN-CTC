#! /bin/bash

#IDAR2013
# train_path='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/ICDAR_2013'
# val_path='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/ICDAR_2013_test'

#IIIT5k
# train_path='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/IIIT5k_Train'
# val_path='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/IIIT5k_Test'

#Synth90k
train_path='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/Synth90k_Train'
val_path='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/Synth90k_Test'

alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

pre_model='/home/wangtongtong/experiment/CRNN_CTC_ICDAR/expr/netCRNN_49.pth'

python train.py --trainRoot ${train_path} --valRoot ${val_path} \
    --cuda --alphabet ${alphabet} --batchSize 256 --workers 2 \
    --lr 0.001 --nepoch 100 --adadelta --displayInterval 1000 \
    # --test --pretrained ${pre_model} \
    >> log.log
