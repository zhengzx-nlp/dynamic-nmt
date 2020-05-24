#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

mkdir -p results

MODEL_NAME="`cat ./save/MODEL_NAME`"
MODEL_PATH=./save/$MODEL_NAME.best.final
CONFIG_PATH=./save/$MODEL_NAME.config
SOURCE_PATH="/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test"
TEST_ALL_PATH="/home/public_data/nmtdata/nist_zh-en_1.34m//test/test_all.ref"
SAVETO=./results/

for i in 3 4 5 6;
do
    src=$SOURCE_PATH/mt0$i.src
    ref=$SOURCE_PATH/mt0$i.ref
    declare -a refs=($SOURCE_PATH/mt0$i.ref0 $SOURCE_PATH/mt0$i.ref1 $SOURCE_PATH/mt0$i.ref2 $SOURCE_PATH/mt0$i.ref3)
    saveto=$SAVETO/mt0$i.trans

    python -m src.bin.translate \
        --source_path $src \
        --model_path $MODEL_PATH \
        --config_path $CONFIG_PATH \
        --saveto $saveto \
        --batch_size 150 \
        --beam_size 5 \
        --alpha 0.6 \
        --keep_n 1 \
        --use_gpu

    BLEU=`sacrebleu --tokenize none -lc ${refs[@]} < $saveto.0`
    results[$i]=$saveto.0
    echo $BLEU
    echo "$saveto.0:    $BLEU" >> ./results/bleu.log
done

cat ${results[4]} ${results[5]} ${results[6]} > $SAVETO/test_all.trans
BLEU=`perl ./src/metric/scripts/generic/multi-bleu.perl -lc $TEST_ALL_PATH < $SAVETO/test_all.trans `
echo $BLEU
echo "$SAVETO/test_all.trans:    $BLEU" >> ./results/bleu.log

