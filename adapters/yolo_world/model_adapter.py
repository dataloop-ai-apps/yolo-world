import os
import logging
import dtlpy as dl

from PIL import Image
from ultralytics import YOLOWorld


logger = logging.getLogger('YOLO-WorldAdapter')

# set max image size
Image.MAX_IMAGE_PIXELS = 933120000

DEFAULT_LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


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
        if self.model_entity.labels is None:
            self.model_entity.labels = self.configuration.get('labels', DEFAULT_LABELS)
        self.dataset_ontologies_lookup = {}

    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        image = Image.open(filename)
        # Check if the image has EXIF data
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()
            # Get the EXIF orientation tag (if available)
            if exif_data is not None:
                orientation = exif_data.get(0x0112)
                if orientation is not None:
                    # Rotate the image based on the orientation tag
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        image = image.convert('RGB')
        if item.dataset.id not in self.dataset_ontologies_lookup:
            labels_entity = item.dataset.ontologies.list()[0].labels
            labels = [label.tag for label in labels_entity]
        if len(labels) == 0:
            labels = DEFAULT_LABELS
        image_dict = {'image': image, 'labels': labels}
        return image_dict

    # def predict_items(self, items: list, upload_annotations=None, clean_annotations=None, batch_size=None, **kwargs):
    #     """
    #     Run the predict function on the input list of items (or single) and return the items and the predictions.
    #     Each prediction is by the model output type (package.output_type) and model_info in the metadata
    #
    #     :param items: `List[dl.Item]` list of items to predict
    #     :param upload_annotations: `bool` uploads the predictions on the given items
    #     :param clean_annotations: `bool` deletes previous model annotations (predictions) before uploading new ones
    #     :param batch_size: `int` size of batch to run a single inference
    #
    #     :return: `List[dl.Item]`, `List[List[dl.Annotation]]`
    #     """
    #     if batch_size is None:
    #         batch_size = self.configuration.get('batch_size', 4)
    #     upload_annotations = self.adapter_defaults.resolve("upload_annotations", upload_annotations)
    #     clean_annotations = self.adapter_defaults.resolve("clean_annotations", clean_annotations)
    #     input_type = self.model_entity.input_type
    #     self.logger.debug(
    #         "Predicting {} items, using batch size {}. input type: {}".format(len(items), batch_size, input_type))
    #     pool = ThreadPoolExecutor(max_workers=16)
    #
    #     annotations = list()
    #     for i_batch in tqdm.tqdm(range(0, len(items), batch_size), desc='predicting', unit='bt', leave=None,
    #                              file=sys.stdout):
    #         batch_items = items[i_batch: i_batch + batch_size]
    #         batch = list(pool.map(self.prepare_item_func, batch_items))
    #         batch_collections = self.predict(batch, **kwargs)
    #         _futures = list(pool.map(partial(self._update_predictions_metadata),
    #                                  batch_items,
    #                                  batch_collections))
    #         # Loop over the futures to make sure they are all done to avoid race conditions
    #         _ = [_f for _f in _futures]
    #         if upload_annotations is True:
    #             self.logger.debug(
    #                 "Uploading items' annotation for model {!r}.".format(self.model_entity.name))
    #             try:
    #                 batch_collections = list(pool.map(partial(self._upload_model_annotations,
    #                                                           clean_annotations=clean_annotations),
    #                                                   batch_items,
    #                                                   batch_collections))
    #             except Exception as err:
    #                 self.logger.exception("Failed to upload annotations items.")
    #
    #         for collection in batch_collections:
    #             # function needs to return `List[List[dl.Annotation]]`
    #             # convert annotation collection to a list of dl.Annotation for each batch
    #             if isinstance(collection, dl.AnnotationCollection):
    #                 annotations.extend([annotation for annotation in collection.annotations])
    #             else:
    #                 logger.warning(f'RETURN TYPE MAY BE INVALID: {type(collection)}')
    #                 annotations.extend(collection)
    #         # TODO call the callback
    #
    #     pool.shutdown()
    #     return items, annotations
    #

    def predict(self, batch, **kwargs):
        for item in batch:
            image = item['image']
            labels = item['labels']
            self.model.set_classes(classes=labels)
            results = self.model.predict(source=image, save=False, save_txt=False)

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
