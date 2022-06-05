FROM nvcr.io/nvidia/tensorflow:22.05-tf2-py3
ARG MYUSER=$USER
ARG MYUID="$(id -u)"
ARG MYGID="$(id -g)"
 
ENV DEBIAN_FRONTEND noninteractive
 
# Install cmake, add user
RUN mkdir -p /workspace && cd /workspace && \
    if [ -f /usr/bin/yum ]; then yum install -y make wget; fi && \
    if [ -f /usr/bin/apt-get ]; then apt-get update && apt-get install -y apt-utils make wget vim; fi && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.23.0/cmake-3.23.0-linux-x86_64.tar.gz && \
    tar xf cmake-3.23.0-linux-x86_64.tar.gz && rm cmake-3.23.0-linux-x86_64.tar.gz && \
    groupadd -f -g ${MYGID} ${MYUSER} && \
    useradd -rm -u $MYUID -g $MYUSER -p "" $MYUSER && \
    chown ${MYUSER}:${MYGID} /workspace
 
USER $MYUSER
ENV PATH /workspace/cmake-3.23.0-linux-x86_64/bin:$PATH
 
WORKDIR /workspace