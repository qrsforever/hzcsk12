FROM hzcsai_com/k12ai

LABEL maintainer="k12rl@hzcsai.com"

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
      org.label-schema.description="CV Backend" \
      org.label-schema.url="https://www.hzcsai.com/index.php?r=front" \
      org.label-schema.vcs-url=$URL \
      org.label-schema.vcs-ref=$COMMIT \
      org.label-schema.vcs-branch=$BRANCH \
      org.label-schema.vendor=$VENDOR \
      org.label-schema.version=$VERSION \
      org.label-schema.docker.cmd="main.py"

WORKDIR /hzcsk12/rl

COPY app app
COPY rlpyt/rlpyt rlpyt
COPY rlpyt/setup.py setup.py
COPY rlpyt/README.md README.md

ENV PATH=/hzcsk12/rl/app/k12ai:$PATH
ENV PYTHONPATH=/hzcsk12/rl/app:/hzcsk12/rl/rlpyt:$PYTHONPATH

RUN PIP_INSTALL="pip install -U --no-cache-dir --retries 20 --timeout 120 \
        --trusted-host mirrors.intra.didiyun.com \
        --index-url http://mirrors.intra.didiyun.com/pip/simple" && \
    $PIP_INSTALL \
        pyprind \
        atari-py gym[atari] gym[box2d] \
        && \
    pip install --editable .

CMD ["/bin/bash"]
