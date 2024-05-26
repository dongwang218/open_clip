#!/bin/bash -x

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m training.main \
    --save-frequency 1 \
    --model "mobilenetv2_050" --name "ft_mobilenetv2_050" \
    --pretrained logs/mobilenetv2_050/checkpoints/epoch_199.pt \
    --resume 'latest' \
    --finetune \
    --train-data=ImageNet  \
    --epochs=3  \
    --lr=0.00003  \
    --batch-size=512  \
    --eval-datasets=ImageNet,ImageNetA  \
    --template=openai_imagenet_template  \
    --results-db=results.jsonl  \
    --data-location=/checkpoint/dongwang/datasets \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
