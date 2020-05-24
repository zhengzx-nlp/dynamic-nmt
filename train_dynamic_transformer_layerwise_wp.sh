#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME="dynamic_transformer-nist_zh2en_bpe-contextual-layerwise_wp"
CONFIG_PATH=./configs/dynamic_transformer_layerwise_nist_zh2en_bpe.yaml
LOG_PATH=/home/user_data55/zhengzx/.pytorch.log/${MODEL_NAME}
SAVETO=./save/
PRETRAIN_MODEL=/home/user_data55/zhengzx/models/nist_transformer_base/save/transformer-nist-zh2en.best.tpz

mkdir -p $SAVETO
cp $CONFIG_PATH $SAVETO/$MODEL_NAME.config

python -m src.bin.train \
    --model_name ${MODEL_NAME} \
    --config_path ${CONFIG_PATH} \
    --log_path ${LOG_PATH} \
    --saveto ${SAVETO} \
    --reload \
    --pretrain_path ${PRETRAIN_MODEL} \
    --use_gpu