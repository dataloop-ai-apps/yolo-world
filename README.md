 YOLO-World Model Adapter

## Introduction

This repo is a model integration between [Ultralytics YOLO-World](https://github.com/ultralytics/ultralytics) model and [Dataloop](https://dataloop.ai/)

YOLO-World builds on YOLOv8's speed and provides zero-shot detection of unknown objects.

## Requirements

- dtlpy
- torch==2.4.1
- ultralytics==8.2.91
- PyYAML==6.0.2
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the YOLO-World model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the
Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory), tags or you can use DQL filter to have training and validation subsets.

### Editing the configuration

To edit configurations via the platform, go to the YOLO-World page in the Model Management and edit the json
file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more
information.

The basic configurations included are:

- `labels`: The labels over which the model will predict (defaults to the labels in the model's dataset's)

## Sources and Further Reading

- [Ultralytics documentation](https://docs.ultralytics.com/models/yolo-world/)

## Acknowledgements

The original YOLO-World paper can be found on [arXiv](https://arxiv.org/abs/2401.17270). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/AILab-CVC/YOLO-World). We appreciate their efforts in advancing the field and making their work accessible to the broader community.