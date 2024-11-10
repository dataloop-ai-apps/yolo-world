from ultralytics import YOLOWorld
from PIL import Image

import dtlpy as dl
import logging
import torch
import PIL
import os

logger = logging.getLogger('YOLO-WorldAdapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000


@dl.Package.decorators.module(description='Model Adapter for YOLO-World object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    """
    Model Adapter class for loading and using the YOLOWorld model.
    """
    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', 'yolov8s-worldv2.pt')
        model_filepath = os.path.join(local_path, model_filename)

        if os.path.isfile(model_filepath):
            model = YOLOWorld(model_filepath)  # pass any model type
        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/' + model_filename
            model = YOLOWorld(url)  # pass any model type
        self.model = model
        
        custom_labels = self.configuration.get('labels', None)
        if custom_labels:
            self.model.set_classes(custom_labels)
            logger.info('Using the following custom labels provided in config: {}'.format(custom_labels))
        else:
            logger.warning('No custom labels provided, using default model labels')
            
    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        if 'image' in item.mimetype:
            data = Image.open(filename)
            # Check if the image has EXIF data
            if hasattr(data, '_getexif'):
                exif_data = data._getexif()
                # Get the EXIF orientation tag (if available)
                if exif_data is not None:
                    orientation = exif_data.get(0x0112)
                    if orientation is not None:
                        # Rotate the image based on the orientation tag
                        if orientation == 3:
                            data = data.rotate(180, expand=True)
                        elif orientation == 6:
                            data = data.rotate(270, expand=True)
                        elif orientation == 8:
                            data = data.rotate(90, expand=True)
            data = data.convert('RGB')
        else:
            data = filename
        return data, item

    def predict(self, batch, **kwargs):
        filtered_streams = list()
        for stream, item in batch:
            if 'image' in item.mimetype:
                filtered_streams.append(stream)
            else:
                logger.warning(f'Item {item.id} mimetype is not supported. Skipping item prediction')

        device = self.configuration.get('device', None)
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        results = self.model.predict(source=filtered_streams, save=False, save_txt=False, device=device)  # save predictions as labels
        batch_annotations = list()
        for _, res in enumerate(results):  # per image
            image_annotations = dl.AnnotationCollection()
            for d in reversed(res.boxes):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                c = int(cls)
                label = res.names[c]
                xyxy = d.xyxy.squeeze()
                image_annotations.add(annotation_definition=dl.Box(left=float(xyxy[0]),
                                                                   top=float(xyxy[1]),
                                                                   right=float(xyxy[2]),
                                                                   bottom=float(xyxy[3]),
                                                                   label=label
                                                                   ),
                                      model_info={'name': self.model_entity.name,
                                                  'model_id': self.model_entity.id,
                                                  'confidence': float(conf)})
            batch_annotations.append(image_annotations)
        return batch_annotations