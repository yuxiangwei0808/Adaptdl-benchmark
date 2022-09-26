# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": []}' -- python main.py --num_worker=2 --batch_size=32 --data_path /data/cifar-10


 