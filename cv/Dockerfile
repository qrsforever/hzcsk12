FROM hzcsai_com/k12cv-base

LABEL maintainer="framework@hzcsai.com"

ARG VENDOR
ARG PROJECT
ARG REPOSITORY
ARG TAG
ARG DATE
ARG VERSION
ARG URL
ARG COMMIT
ARG BRANCH
ARG PORT

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
      org.label-schema.docker.cmd="docker run -d --name $PROJECT \
--restart unless-stopped --network host --hostname $PROJECT \
--volume /data:/data --volume /data/pretrained:/root/.torch/models \
--runtime nvidia --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
$REPOSITORY:$TAG python app_service.py --port $PORT"

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY apis apis/
COPY cauchy cauchy/
COPY setup.py setup.py

# install vulkan
RUN cd cauchy/utils/vulkan/ && \
        python setup.py install

# install coco api
RUN cd apis/cocoapi/PythonAPI/ && \
        python setup.py install

# install torchcv extensions
RUN cd cauchy/extensions/ && \
        chmod +x make.sh && \
        sh make.sh

# install torchcv
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

RUN pip install --editable .

CMD ["/bin/bash"]

WORKDIR /hzcsk12/cv/app
