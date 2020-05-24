#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p results

MODEL_NAME=""
MODEL_PATH=./save/$MODEL_NAME.best.final
CONFIG_PATH=./save/$MODEL_NAME.config
SOURCE_PATH="/home/user_data/zhengzx/data/domain_monolingual/arxiv_cs/trans_test.zh"
REFERENCE_PATH_PREFIX="/home/user_data/zhengzx/data/domain_monolingual/arxiv_cs/trans_test.en"
SAVETO=./results/arxiv_cs-trans_test.trans.en

python -m src.bin.translate \
    --source_path $SOURCE_PATH \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --saveto $SAVETO \
    --batch_size 25 \
    --beam_size 5 \
    --alpha 0.6 \
    --keep_n 1 \
    --use_gpu

BLEU=`sacrebleu --tokenize none -lc $REFERENCE_PATH_PREFIX < $SAVETO.0`
echo $BLEU
echo "$SAVETO.0:    $BLEU" >> ./results/bleu.log
