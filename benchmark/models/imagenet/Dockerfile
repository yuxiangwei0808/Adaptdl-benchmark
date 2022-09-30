FROM nvcr.io/nvidia/pytorch:21.10-py3
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
