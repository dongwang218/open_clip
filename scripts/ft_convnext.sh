#!/bin/bash -x

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m training.main \
    --save-frequency 1 \
    --model "convnext_base_w" --name "ft_convnext_256" \
    --pretrained "laion2b_s13b_b82k_augreg" \
    --resume 'latest' \
    --finetune \
    --train-data=ImageNet  \
    --epochs=3  \
    --lr=0.00003  \
    --batch-size=64  \
    --eval-datasets=ImageNet,ImageNetA  \
    --template=openai_imagenet_template  \
    --results-db=results.jsonl  \
    --data-location=/checkpoint/dongwang/datasets \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
