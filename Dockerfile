## EfficientNet の Dockerfile
## GPU での学習・推論用
##
## イメージのビルド
##   $ docker build -t shapemask
## コンテナ構築
##   $ docker run -itdv /path/to/cloned/repo:/work --name shapemask shapemask bash
## 学習
##   $ docker exec -it shapemask bash

FROM tensorflow/tensorflow:1.15.3-gpu-py3

## まじない
RUN apt-get update && apt-get install -y libtcmalloc-minimal4

## Google Cloud SDK (gsutil)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
    && apt-get update -y \
    && apt-get install google-cloud-sdk -y

## Tensorflow 関連
RUN pip install --upgrade pip
COPY ./requirements.txt /tmp/shapemask/
RUN pip install -r /tmp/shapemask/requirements.txt

## tensorflow/models（データセット作成時に使用） 
WORKDIR /work/
RUN apt-get install -y git protobuf-compiler
RUN git clone --depth=1 https://github.com/tensorflow/models tensorflow_models
WORKDIR /work/tensorflow_models/
RUN ["/bin/bash", "-c", "cd /work/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=."]
ENV PYTHONPATH "${PYTHONPATH}:/work/tensorflow_models:/work/tensorflow_models/research/:/work/tensorflow_models/research/slim:/work/repo"

CMD bash
