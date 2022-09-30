FROM nvcr.io/nvidia/pytorch:21.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/bert-sst2/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/bert-sst2/transformers transformers
RUN cd transformers; pip install .

COPY benchmark/models/bert-sst2/file_download.py /opt/conda/lib/python3.8/site-packages/huggingface_hub/file_download.py
COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/bert-sst2 bert-sst2
ENV PYTHONPATH=/root/bert-sst2:$PYTHONPATH
WORKDIR /root/bert-sst2

ENV PYTHONUNBUFFERED=true
