#!/bin/bash -x

export PYTHONPATH="$PYTHONPATH:$PWD/src"

rm -rf logs/ft_convnext_vww
python -m training.main \
    --save-frequency 1 \
    --model "convnext_base_w" --name "ft_convnext_vww" \
    --pretrained "laion2b_s13b_b82k_augreg" \
    --finetune \
    --train-data=VWW  \
    --epochs=1  \
    --lr=0.00003  \
    --batch-size=256  \
    --workers=24 \
    --eval-datasets=VWW  \
    --template=openai_imagenet_template  \
    --results-db=results.jsonl  \
    --data-location=/checkpoint/dongwang/datasets \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --classnames=vww \
    --bias=-0.2341