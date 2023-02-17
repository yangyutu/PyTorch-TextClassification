export CUDA_VISIBLE_DEVICES="0"
pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
#pretrained_model_name="bert-base-uncased"
dataset_names=('YelpReviewFull' 'YelpReviewPolarity')
dataset_names=( 'YahooAnswers' 'AG_NEWS' )

for name in "${dataset_names[@]}"
do
    python text_classification/run_pretrained_transformerV2.py \
    --dataset_name ${name} \
    --pretrained_model_name ${pretrained_model_name} \
    --gpus 1 \
    --max_epochs 10 \
    --lr 3e-6 \
    --truncate 256 \
    --batch_size 128 \
    --num_workers 16 \
    --project_name text_classification \
    --default_root_dir ./experiments/logs
done