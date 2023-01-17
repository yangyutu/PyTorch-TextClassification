#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0"

dataset_names=('AG_NEWS' 'AmazonReviewFull' 'AmazonReviewPolarity' 'YelpReviewFull' 'YahooAnswers')
dataset_names=('AG_NEWS' 'YelpReviewFull' 'YelpReviewPolarity' 'YahooAnswers')
dataset_names=('YelpReviewPolarity')

model_name=logistic_bag
for name in "${dataset_names[@]}"
do
    python text_classification/run_logistic_bag.py \
    --dataset_name ${name} \
    --gpus 1 \
    --max_epochs 10 \
    --lr 1e-2 \
    --batch_size 256 \
    --num_workers 16 \
    --project_name text_classification \
    --model_name ${model_name} \
    --default_root_dir ./experiments/logs
done