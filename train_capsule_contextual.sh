#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME="capsule-nist_zh2en_bpe-contextual"
CONFIG_PATH=./configs/capsule_nist_zh2en_bpe_contextual.yaml
LOG_PATH=/home/user_data/zhengzx/.pytorch.log/${MODEL_NAME}
SAVETO=./save/
PRETRAIN_MODEL=/home/user_data/zhengzx/models/mt/nist_zh2en_bpe_hybrid_gru_attn_zeroinit_base/save/hybrid-nist_zh2en_bpe-base-gru_attn_zero.best.final

mkdir -p $SAVETO
cp $CONFIG_PATH $SAVETO/$MODEL_NAME.config

python -m src.bin.train \
    --model_name ${MODEL_NAME} \
    --config_path ${CONFIG_PATH} \
    --log_path ${LOG_PATH} \
    --saveto ${SAVETO} \
    --pretrain_path ${PRETRAIN_MODEL} \
    --reload \
    --use_gpu