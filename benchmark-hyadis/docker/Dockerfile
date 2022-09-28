FROM cr-cn-beijing.volces.com/hpcaitech/hyadis-runtime:latest

RUN /bin/sh -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
apt-get update && \
apt-get -y --force-yes install libgl1-mesa-glx libglib2.0-dev"

COPY ./apex /tmp/apex
RUN /bin/sh -c "cd /tmp/apex && python setup.py install"

