# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": ["numpy", "pandas", "scipy"]}' \
-- \
python main.py \
--epoch=20 \
--num_workers=4 \
--learning_rate=0.001 \
--batch_size=4
 