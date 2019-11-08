FROM hzcsai_com/waltzcv-base

LABEL maintainer="framework@hzcsai.com"

ARG VENDOR="hzcsai"
ARG REPOSITORY
ARG TAG
ARG DATE
ARG VERSION
ARG URL
ARG COMMIT
ARG BRANCH

LABEL org.label-schema.schema-version="1.0" \
      org.label-schema.build-date=$DATE \
      org.label-schema.name=$REPOSITORY \
      org.label-schema.description="CV Backend" \
      org.label-schema.url=https://www.hzcsai.com/index.php?r=front \
      org.label-schema.vcs-url=$URL \
      org.label-schema.vcs-ref=$COMMIT \
      org.label-schema.vcs-branch=$BRANCH \
      org.label-schema.vendor=$VENDOR \
      org.label-schema.version=$VERSION \
      org.label-schema.docker.cmd="docker run -d --name waltzcv \
--restart unless-stopped --network host --hostname waltzcv \
--volume /data:/data --volume /data/pretrained:/root/.torch/models \
--runtime nvidia --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
$REPOSITORY:$TAG python cauchy_services.py --port 8339"

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /Softwares/torchcv/

COPY apis /Softwares/torchcv/apis/
COPY cauchy /Softwares/torchcv/cauchy/
COPY setup.py LICENSE MANIFEST.in README.md /Softwares/torchcv/

# install vulkan
RUN cd /Softwares/torchcv/cauchy/utils/vulkan/ && \
        python setup.py install

# install coco api
RUN cd /Softwares/torchcv/apis/cocoapi/PythonAPI/ && \
        python setup.py install

# install torchcv extensions
RUN cd /Softwares/torchcv/cauchy/extensions/ && \
        chmod +x make.sh && \
        sh make.sh

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# install torchcv
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN cd /Softwares/torchcv/ && \
        python setup.py install

RUN rm -rf /Softwares

RUN mkdir /app

COPY test/flask_services /app

WORKDIR /app
