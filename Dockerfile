FROM caipz/zhijianlifekpistaskone:version1
COPY ./model /model
COPY ./process /process
COPY ./input /input
COPY ./output /output

## Install Python packages in Docker image

RUN pip3 install --upgrade pip


RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list \
    && sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list


RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
        python3-pip \
        python3-dev \
        gcc \
        g++ \
        openslide-tools\
        libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install Openslide-python -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip3 install imageio -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip3 install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip3 install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple 
CMD ["../model/inference_docker.py"]
ENTRYPOINT ["python3"]




