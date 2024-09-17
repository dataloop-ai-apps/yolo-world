FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

COPY requirements.txt /requirements.txt

RUN pip install --user -r /requirements.txt