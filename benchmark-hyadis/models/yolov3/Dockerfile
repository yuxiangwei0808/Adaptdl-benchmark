FROM cr-cn-beijing.volces.com/hpcaitech/adaptdl-submit:21.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/yolov3/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update; apt-get -y --force-yes install libgl1-mesa-glx; apt-get -y --force-yes install libglib2.0-dev
COPY benchmark/models/imagenet/apex apex
RUN cd apex; python setup.py install

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/yolov3 yolov3
ENV PYTHONPATH=/root/yolov3:$PYTHONPATH
WORKDIR /root/yolov3

ENV PYTHONUNBUFFERED=true
