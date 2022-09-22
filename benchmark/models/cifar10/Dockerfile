FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime-cifar10-imagenet1k
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/cifar10/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/cifar10 cifar10
ENV PYTHONPATH=/root/cifar10:$PYTHONPATH
WORKDIR /root/cifar10

ENV PYTHONUNBUFFERED=true
