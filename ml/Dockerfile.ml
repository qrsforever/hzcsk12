FROM hzcsai_com/k12ai

LABEL maintainer="k12ml@hzcsai.com"

ARG VENDOR
ARG PROJECT
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
      org.label-schema.description="ML Backend" \
      org.label-schema.url="https://www.hzcsai.com/index.php?r=front" \
      org.label-schema.vcs-url=$URL \
      org.label-schema.vcs-ref=$COMMIT \
      org.label-schema.vcs-branch=$BRANCH \
      org.label-schema.vendor=$VENDOR \
      org.label-schema.version=$VERSION \
      org.label-schema.docker.cmd="TODO"

WORKDIR /hzcsk12/ml

COPY app app

ENV PYTHONPATH=/hzcsk12/ml/app:/hzcsk12/ml/mla:$PYTHONPATH

CMD ["/bin/bash"]
