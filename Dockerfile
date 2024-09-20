FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

RUN pip install --user torch==2.4.1 ultralytics==8.2.91 dtlpy PyYAML==6.0.2 pytest