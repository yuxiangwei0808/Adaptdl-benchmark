# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": []}' -- \

python3 run_glue.py \
--model_name_or_path=bert-base-uncased \
--task_name=sst2 \
--data_dir=$HOME/tmp/data \
--per_device_train_batch_size=4 \
--learning_rate=3e-5 \
--num_train_epochs=3 \
--cache_dir=$HOME/tmp/glue \
--output_dir=$HOME/tmp/output \
--with_tracking

 