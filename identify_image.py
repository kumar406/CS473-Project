import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np

THRESHOLD = 0.3

def load_model(pipeline_config_path, ckpt_path):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(ckpt_path).expect_partial()
    return detection_model

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def label_image(img_path, label_map_path, detection_model):
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
    img = cv2.imread(img_path)
    image_np = np.array(img)
    y_h = image_np.shape[0]
    x_h = image_np.shape[1]

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    final_detections = []
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] > THRESHOLD:
            dbox = np.copy(detections['detection_boxes'][i])
            dbox[0] *= y_h
            dbox[1] *= x_h
            dbox[2] *= y_h
            dbox[3] *= x_h
            final_detections.append({
                'label': category_index[detections['detection_classes'][i] + label_id_offset],
                'box_norm': detections['detection_boxes'][i],
                'box': dbox.astype(int),
                'score': detections['detection_scores'][i]
                })
    return final_detections
def label_images(img_paths, label_map_path, detection_model):
    labeled_images = []
    for img_path in img_paths:
        labeled_images.append({
            'img': img_path,
            'labels': label_image(img_path, label_map_path, detection_model)
        })
    return labeled_images
