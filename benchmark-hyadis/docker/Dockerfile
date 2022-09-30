FROM cr-cn-beijing.volces.com/hpcaitech/hyadis-runtime:latest

RUN /bin/sh -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
apt-get update && \
apt-get -y --force-yes install libgl1-mesa-glx libglib2.0-dev"

COPY ./apex /tmp/apex
RUN /bin/sh -c "cd /tmp/apex && python setup.py install"

ADD ./requirements.txt /tmp/requirements.txt
RUN /bin/sh -c "pip install -r /tmp/requirements.txt"

RUN /bin/sh -c "cp -t /opt/conda/lib/python3.7/site-packages/hyadis/core/bin \
/opt/conda/lib/python3.7/site-packages/ray/core/src/ray/gcs/gcs_server \
/opt/conda/lib/python3.7/site-packages/ray/core/src/ray/raylet/raylet"