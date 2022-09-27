# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": ["numpy", "pandas", "gensim", "tensorboardX", "tensorboard"]}' -- python main.py
 