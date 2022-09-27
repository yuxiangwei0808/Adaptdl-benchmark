# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": [ "torchvision", "tensorboard", "tensorboardX" ]}' -- \
python main.py \
        --arch=resnet50 \
        --batch-size=4 \
        --lr=0.01 \
        --epochs=150 \
        --autoscale-bsz \
        /data/imagenet-1k/


# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# python setup.py install
 