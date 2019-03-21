FROM speech2face

#ENV LANG en_US.UTF-8
#ENV LANGUAGE en_US:en
#ENV LC_ALL en_US.UTF-8

#RUN apt-get update
#RUN apt-get install -y curl locales git python3.6 python3.6-dev python3-pip
#RUN locale-gen en_US.UTF-8
RUN mkdir /glow
WORKDIR /glow

#RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
#RUN pip3 install torchvision tqdm scipy

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
#RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh

#ENV PATH="/opt/conda/bin:${PATH}"

#RUN conda install -y python=3.6 pytorch torchvision cudatoolkit=10.0 -c pytorch

