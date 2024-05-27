#!/bin/bash -x

export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m training.main \
    --save-frequency 1 \
    --model "convnext_base_w" --name "ft_convnext_vww" \
    --pretrained "laion2b_s13b_b82k_augreg" \
    --finetune \
    --train-data=VWW  \
    --epochs=1  \
    --lr=0.00003  \
    --batch-size=64  \
    --eval-datasets=VWW  \
    --template=openai_imagenet_template  \
    --results-db=results.jsonl  \
    --data-location=/checkpoint/dongwang/datasets \
    --alpha 0 1.0 \
    --classnames=vww \
    --load=logs/ft_convnext_vww/zeroshot.pt,./logs/ft_convnext_vww/finetuned/checkpoint_1.pt 
