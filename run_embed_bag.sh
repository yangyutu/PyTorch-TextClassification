#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0"

dataset_names=('AG_NEWS' 'AmazonReviewFull' 'AmazonReviewPolarity' 'SogouNews')
dataset_names=('YelpReviewFull' 'YahooAnswers')

model_name=embedding_bag
for name in "${dataset_names[@]}"
do
    python text_classification/run_embed_bag.py \
    --dataset_name ${name} \
    --gpus 1 \
    --max_epochs 3 \
    --lr 1e-3 \
    --batch_size 256 \
    --num_workers 16 \
    --project_name text_classification \
    --model_name ${model_name} \
    --default_root_dir ./experiments/logs
done