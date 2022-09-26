# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": [ "torchvision", "tensorboard", "tensorboardX" ]}' \
-- python main.py \
        --arch=resnet50 \
        --batch-size=128 \
        --lr=0.01 \
        --epochs=150 \
        --autoscale-bsz \
        /data/scratch/imagenet-100/



 