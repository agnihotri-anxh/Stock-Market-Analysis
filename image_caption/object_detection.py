import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st

def simple_object_detection(image, threshold=0.5):
    """
    Simple object detection using TensorFlow Hub's pre-trained model
    Args:
        image: Input image (numpy array or PIL Image)
        threshold: Confidence threshold (0-1)
    Returns:
        image_with_boxes: Image with detected objects
        detections: List of detected objects (label, confidence, bounding box)
    """
    # Load model (cached after first load)
    if not hasattr(simple_object_detection, 'model'):
        simple_object_detection.model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
    # Convert image to numpy array if needed
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    # Convert to tensor and run detection
    input_tensor = tf.convert_to_tensor(img_np[np.newaxis, ...], dtype=tf.uint8)
    results = simple_object_detection.model(input_tensor)
    # Process results
    boxes = results['detection_boxes'][0].numpy()
    scores = results['detection_scores'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(int)
    # COCO labels (81 classes)
    labels = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    # Filter detections by threshold
    detections = []
    image_with_boxes = img_np.copy()
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right = int(xmin * w), int(xmax * w)
            top, bottom = int(ymin * h), int(ymax * h)
            label = labels[classes[i]]
            confidence = float(scores[i])
            # Store detection info
            detections.append({
                'label': label,
                'confidence': confidence,
                'box': [left, top, right, bottom]
            })
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image_with_boxes, (left, top), (right, bottom), color, 2)
            cv2.putText(image_with_boxes, 
                       f"{label} {confidence:.2f}", 
                       (left, max(top-10, 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
    return image_with_boxes, detections

def non_max_suppression(boxes, scores, threshold=0.5):
    # boxes: [N, 4] in (ymin, xmin, ymax, xmax) normalized
    # scores: [N]
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep

def detect_objects(img, detector, threshold=0.3, nms_threshold=0.5, filter_classes=None):
    img_resized = img.resize((320, 320))
    img_np = np.array(img_resized).astype(np.uint8)
    input_tensor = tf.convert_to_tensor(img_np[None, ...], dtype=tf.uint8)
    try:
        result = detector(input_tensor)
        if isinstance(result, tuple):
            boxes, scores, classes, num = result
            result = {
                'detection_boxes': boxes.numpy()[0],
                'detection_scores': scores.numpy()[0],
                'detection_classes': classes.numpy()[0]
            }
        else:
            result = {key:value.numpy() for key,value in result.items()}
    except Exception as e:
        st.error(f"Object detection error: {e}")
        return {}, []
    coco_labels = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    boxes, scores, classes = result['detection_boxes'], result['detection_scores'], result['detection_classes']
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    # Filter by threshold
    mask = scores >= threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    # NMS
    keep = non_max_suppression(boxes, scores, threshold=nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    # Class filtering
    detected = []
    detection_dicts = []
    for i in range(boxes.shape[0]):
        class_id = int(classes[i])
        label = coco_labels[class_id] if class_id < len(coco_labels) else str(class_id)
        if filter_classes and label not in filter_classes:
            continue
        detected.append(label)
        detection_dicts.append({
            'label': label,
            'score': float(scores[i]),
            'box': boxes[i].tolist()
        })
    # Return both the raw result and the detection dicts
    return result, detected, detection_dicts

def draw_boxes(img, result, threshold=0.3, mode='square'):
    img_np = np.array(img)
    h, w, _ = img_np.shape
    boxes, scores, classes = result['detection_boxes'], result['detection_scores'], result['detection_classes']
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    coco_labels = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    color_map = {}
    for i in range(boxes.shape[0]):
        score = scores[i]
        if isinstance(score, (np.ndarray, list)):
            score = float(np.squeeze(score))
        if score < threshold:
            continue
        box = boxes[i]
        class_id = int(classes[i])
        label = coco_labels[class_id] if class_id < len(coco_labels) else str(class_id)
        if label not in color_map:
            color_map[label] = tuple([np.random.randint(0,255) for _ in range(3)])
        color = color_map[label]
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
        if mode == 'square':
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            side = max(right - left, bottom - top)
            half_side = side // 2
            sq_left = max(cx - half_side, 0)
            sq_right = min(cx + half_side, w-1)
            sq_top = max(cy - half_side, 0)
            sq_bottom = min(cy + half_side, h-1)
            img_np = cv2.rectangle(img_np, (sq_left, sq_top), (sq_right, sq_bottom), color, 2)
            img_np = cv2.putText(img_np, f"{label} {score:.2f}", (sq_left, max(sq_top-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            img_np = cv2.rectangle(img_np, (left, top), (right, bottom), color, 2)
            img_np = cv2.putText(img_np, f"{label} {score:.2f}", (left, max(top-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_np 