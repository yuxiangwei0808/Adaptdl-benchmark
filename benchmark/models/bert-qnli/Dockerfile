FROM cr-cn-beijing.volces.com/hpcaitech/adaptdl-submit:21.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/bert-qnli/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt; pip install git+https://github.com/huggingface/transformers

COPY benchmark/models/bert-qnli/file_download.py /opt/conda/lib/python3.8/site-packages/huggingface_hub/file_download.py
COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/bert-qnli bert-qnli
ENV PYTHONPATH=/root/bert-qnli:$PYTHONPATH
WORKDIR /root/bert-qnli

ENV PYTHONUNBUFFERED=true
