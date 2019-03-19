FROM speech2face

RUN apt-get install -y tmux vim

RUN mkdir /glow
WORKDIR /glow

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
