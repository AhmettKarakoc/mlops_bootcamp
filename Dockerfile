FROM centos:centos8
#ADD gcp-key.json /gcp-key.json

RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

RUN yum update -y
RUN yum install sudo -y
RUN yum groupinstall "Development Tools" -y
RUN yum install python39 -y
RUN yum install python3-pip -y
RUN pip3 install --upgrade pip

COPY src/ /mlops_bootcamp/src
COPY data/ /mlops_bootcamp/data
COPY models/ /mlops_bootcamp/models
COPY ./requirements.txt /mlops_bootcamp
#COPY ./pyproject.toml /mlops_bootcamp
#COPY ./poetry.lock /mlops_bootcamp
COPY ./ds_app.py /mlops_bootcamp
#RUN mkdir /home/gcp_keys
#COPY gcp-key.json /home/gcp_keys/gcp-key.json

WORKDIR /mlops_bootcamp

RUN pip3 install -r requirements.txt

#RUN pip install poetry
#RUN poetry config virtualenvs.create false
#RUN poetry install --no-dev
