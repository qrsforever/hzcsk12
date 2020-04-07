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

ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
ENV CUDA_HOME=/usr/local/cuda

ADD requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --timeout 120 -r requirements.txt
RUN rm requirements.txt

COPY app app
COPY rlpyt/rlpyt rlpyt
COPY rlpyt/setup.py setup.py
COPY rlpyt/README.md README.md

RUN pip install --editable .

ENV PATH=/hzcsk12/rl/app/k12ai:$PATH
ENV PYTHONPATH=/hzcsk12/rl/app:/hzcsk12/rl/rlpyt:$PYTHONPATH

# ENTRYPOINT ["/bin/bash", "-c", "/usr/bin/xvfb-run -a $@", ""]
CMD ["/bin/bash"]
