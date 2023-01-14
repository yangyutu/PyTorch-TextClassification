export CUDA_VISIBLE_DEVICES="0"
pretrained_model_name="bert-base-uncased"
dataset_names=('YelpReviewFull' 'YelpReviewPolarity' 'YahooAnswers')
#dataset_names=('YahooAnswers')
dataset_names=('AG_NEWS')

for name in "${dataset_names[@]}"
do
    python text_classification/run_custom_transformer.py \
    --dataset_name ${name} \
    --pretrained_tokenizer_name ${pretrained_model_name} \
    --num_layers 2 \
    --nhead 2 \
    --dim_model 128 \
    --dim_feedforward 512 \
    --truncate 512 \
    --pooled_output_embedding \
    --gpus 1 \
    --max_epochs 10 \
    --lr 3e-4 \
    --batch_size 512 \
    --num_workers 16 \
    --project_name text_classification \
    --default_root_dir ./experiments/logs
done