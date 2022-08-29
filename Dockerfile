FROM pytorch/pytorch
#FROM nvcr.io/nvidia/pytorch:22.04-py3


MAINTAINER MikaPommeranz

RUN apt-get update && apt-get install -y git

RUN cd /root \
    && git clone https://github.com/Migga98/MultimodalLearningChess \
    && cd /root/MultimodalLearningChess \
    && pip install -r requirements.txt

CMD cd /root/MultimodalLearningChess/src/ \
    && /bin/bash

#&& git clone https://github.com/Migga98/MultimodalLearningChess.git \