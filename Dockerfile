FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04

WORKDIR /workspace

RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-dev

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install pandas h5py torchsummary ujson opencv-python