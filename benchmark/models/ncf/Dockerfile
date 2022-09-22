FROM cr-cn-beijing.volces.com/hpcaitech/adaptdl-submit:21.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/ncf/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
COPY benchmark/models/ncf/parallel.py /root/adaptdl/adaptdl/torch/parallel.py
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/ncf ncf
ENV PYTHONPATH=/root/ncf:$PYTHONPATH
WORKDIR /root/ncf

ENV PYTHONUNBUFFERED=true
