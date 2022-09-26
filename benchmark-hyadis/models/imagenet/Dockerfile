FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime-cifar10-imagenet1k
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/imagenet/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/imagenet/apex apex
RUN cd apex; python setup.py install

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/imagenet imagenet
ENV PYTHONPATH=/root/imagenet:$PYTHONPATH
WORKDIR /root/imagenet

ENV PYTHONUNBUFFERED=true
