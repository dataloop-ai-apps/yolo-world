import dtlpy as dl
from adapters.yolo_world.model_adapter import Adapter

logger = dl.logger.get_logger(name="YOLOWorldSam2Adapter")


class YOLOWorldSam2Adapter:
    def __init__(self):
        project = dl.projects.get(project_name="DataloopTasks")
        self.sam2_service = project.service.get(service_name="global-sam")
        self.yolo_adapter = Adapter()

    @staticmethod
    def get_labels(item):
        return [label.tag for label in item.dataset.labels]

    def pred_box_and_seg(self, batch, **kwargs):
        labels = YOLOWorldSam2Adapter.get_labels(batch[0])
        images = [self.yolo_adapter.prepare_item_func(item) for item in batch]
        bboxes = self.yolo_adapter.predict(batch=images, labels=labels)
        builder = dl.AnnotationBuilder()
        for bbox in bboxes:
            builder.add_box(label=bbox.label, x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height)

        # get predictions from YOLOWorld
        yolo_predictions = self.yolo_adapter.predict(images, **kwargs)
        logger.info("Finished running YOLOWorld predictions")
        logger.info("Running SAM2 predictions")
        batch_annotations = list()
        for item, annotations in zip(batch, yolo_predictions):
            image_annotations = dl.AnnotationCollection()
            bounding_boxes = [annotation.to_json() for annotation in annotations]
            if not bounding_boxes:
                continue
            sam2_prediction = self.sam2_adapter.sam_predict_box(item=item, annotations=bounding_boxes)

            sam_ann = dl.Annotation.from_json(sam2_prediction[0])
            image_annotations.add(annotation_definition=dl.Polygon(geo=sam_ann.geo, label=sam_ann.label),
                                  model_info={'name': self.model_entity.name,
                                              'model_id': self.model_entity.id,
                                              'confidence': 1},
                                  metadata=annotations[0].metadata)
            batch_annotations.append(image_annotations)
        logger.info("Finished running SAM2 predictions")

        sam2_prediction = sam2_prediction.wait()
        sam2_prediction
