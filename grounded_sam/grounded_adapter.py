import dtlpy as dl
from model_adapter import Adapter as YOLOWorldAdapter
import logging
import base64
import cv2
import numpy as np

logger = logging.getLogger('Grounded-YoloWorld-Sam2-Adapter')
SAM_SERVICE_NAME = "global-sam"


class GroundedSAMAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        super().__init__(model_entity)
        self.yolo_adapter = None
        self.sam2_service = None

    def load(self, local_path, **kwargs):
        """
        Load the wrapped YOLOWorld adapter and the SAM2 service.
        """
        # Initialize YOLOWorldAdapter
        self.yolo_adapter = YOLOWorldAdapter()
        logger.info("YOLOWorld Adapter successfully loaded")

        # Connect to SAM2 Service
        self.sam2_service = dl.services.get(service_name=SAM_SERVICE_NAME)
        if not self.sam2_service:
            raise ValueError(f"SAM Service '{SAM_SERVICE_NAME}' not found")
        logger.info(f"Connected to SAM Service: {self.sam2_service.name}")

    def prepare_item_func(self, item):
        return item

    def predict(self, batch, **kwargs):
        """
        Perform predictions using YOLOWorld and pass bounding boxes to SAM2.
        """
        logger.info("Running YOLOWorld predictions")
        self.yolo_adapter.model.set_classes([label.tag for label in batch[0].dataset.labels])
        images = [self.yolo_adapter.prepare_item_func(item) for item in batch]
        yolo_predictions = self.yolo_adapter.predict(images, **kwargs)

        batch_annotations = list()
        for item, annotations in zip(batch, yolo_predictions):
            image_annotations = dl.AnnotationCollection()
            bounding_boxes = []
            for annotation in annotations:
                bounding_boxes.append(annotation.to_json())

            # Invoke SAM2 Service for segmentation
            logger.info(f"Invoking SAM2 Service for item {item.id}")
            sam2_execution = self.sam2_service.execute(function_name='box_to_segmentation',
                                                       execution_input={'item': item.id,
                                                                        'annotations': bounding_boxes},
                                                       project_id=item.project_id)
            sam2_execution = sam2_execution.wait()
            if sam2_execution.status[-1]['status'] != "success":
                logger.error(f"SAM2 Service failed with error: {sam2_execution.status[-1]['message']}")
                continue
            sam2_output = sam2_execution.output
            logger.info(f"SAM2 Service returned predictions for item {item.id}")

            binary_mask = self.handle_sam_output(self, sam2_output)
            logger.info(f"Mask created from SAM2 predictions")

            image_annotations.add(annotation_definition=dl.Segmentation(geo=binary_mask,
                                                                        label=sam2_output[0]['label']),
                                  model_info={'name': self.model_entity.name,
                                              'model_id': self.model_entity.id,
                                              'confidence': 1},
                                  metadata=sam2_output[0]['metadata'])

            batch_annotations.append(image_annotations)

        return batch_annotations

    @staticmethod
    def handle_sam_output(self, sam_output):
        base64_string = sam_output[0]['coordinates']

        # Remove the prefix 'data:image/png;base64,' if it exists
        prefix = "data:image/png;base64"
        if base64_string.startswith(prefix):
            base64_string = base64_string.replace(prefix, "")

        # Decode the Base64 string to image binary data
        decoded_image_data = base64.b64decode(base64_string)

        # Convert the binary data to a NumPy array
        image_array = np.frombuffer(decoded_image_data, dtype=np.uint8)

        # Decode the NumPy array into an image using OpenCV
        mask = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        # Ensure the mask is successfully loaded
        if mask is None:
            raise ValueError("Failed to decode the mask image from the Base64 string.")

        # Threshold the mask to create a binary image
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return binary_mask
