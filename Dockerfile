FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
RUN sed -i 's/\(archive\|security\).ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /etc/pip.conf

ENV CAFFE_ROOT=/opt/ms-caffe
WORKDIR $CAFFE_ROOT
# build caffe
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libgeos-dev \
        python-opencv \
        python-tk \
        vim \
        thrift-compiler \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN git clone --depth 1 https://github.com/Microsoft/caffe.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done
COPY ./docker/Makefile.config $CAFFE_ROOT
RUN  make -j"$(nproc)" && \
     make pycaffe -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN wget \
https://pypi.python.org/packages/73/9e/fe761e03de28b51b445ddf01ddae87441b7e7040df7d830b86db8f945808/Polygon2-2.0.8.tar.gz#md5=3349a6dfc4cda2a1bcc9bf6c9d411470 \
&& pip install Polygon2-2.0.8.tar.gz && \
rm Polygon2-2.0.8.tar.gz
