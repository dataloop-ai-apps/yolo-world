import dtlpy as dl
from model_adapter import Adapter as YOLOWorldAdapter
import logging
import base64
import cv2
import numpy as np

logger = logging.getLogger('Grounded-YoloWorld-Sam2-Adapter')


class GroundedSAMAdapter(dl.BaseModelAdapter):
    def __init__(self):
        super().__init__()
        self.yolo_adapter = None
        self.sam2_service = None

    def load(self, local_path, **kwargs):
        """
        Load the wrapped YOLOWorld adapter and optionally extend its behavior.
        """

        # Initialize YOLOWorldAdapter with the same model_entity
        self.yolo_adapter = YOLOWorldAdapter(model_entity=self.model_entity)
        # Load the YOLOWorldAdapter with the provided local path
        self.yolo_adapter.load(local_path, **kwargs)
        logger.info("YOLOWorld Adapter successfully loaded")

        # Connect to SAM2 Service
        sam_service_name = "global-sam"
        self.sam2_service = dl.services.get(service_name=sam_service_name)
        if not self.sam2_service:
            raise ValueError(f"SAM Service '{sam_service_name}' not found")
        logger.info(f"Connected to SAM Service: {self.sam2_service.name}")

    def predict(self, batch, **kwargs):
        """
        Perform predictions using YOLOWorld and pass bounding boxes to SAM2.
        """
        logger.info("Running YOLOWorld predictions")
        yolo_predictions = self.yolo_adapter.predict(batch, **kwargs)

        batch_annotations = list()
        for item, annotations in zip(batch, yolo_predictions):
            # Convert YOLOWorld bounding boxes to SAM format
            bounding_boxes = []
            image_annotations = dl.AnnotationCollection()
            for annotation in annotations:
                box = {
                    'coordinates': [
                        {'x': annotation.left, 'y': annotation.top},
                        {'x': annotation.right, 'y': annotation.bottom}
                    ],
                    'label': annotation.label,
                    'attributes': annotation.attributes
                }
                bounding_boxes.append(box)

            # Invoke SAM2 Service for segmentation
            logger.info(f"Invoking SAM2 Service for item {item.id}")
            sam2_execution = self.sam2_service.execute(
                execution_input={'item_id': item.id,
                                 'annotations': bounding_boxes}
            )
            sam2_execution = sam2_execution.wait()
            sam2_output = sam2_execution.output
            logger.info(f"SAM2 Service returned predictions for item {item.id}")

            binary_mask = self.handle_sam_output(self, sam2_output)
            logger.info(f"Mask created from SAM2 predictions")

            image_annotations.add(annotation_definition=dl.Segmentation(geo=binary_mask,
                                                                        label=sam2_output[0]['label']),
                                  model_info={'name': self.model_entity.name,
                                              'model_id': self.model_entity.id,
                                              'confidence': 1},
                                  metadata=sam2_output.metadata)

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
