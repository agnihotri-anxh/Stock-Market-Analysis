import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

def opencv_basic_colored_segmentation(img, morph_kernel=5, alpha=0.4):
    img_np = np.array(img)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    seg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    seg_clean = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)
    seg_clean = cv2.morphologyEx(seg_clean, cv2.MORPH_CLOSE, kernel)
    colored_mask = cv2.applyColorMap(seg_clean, cv2.COLORMAP_JET)
    # Edge highlighting
    edges = cv2.Canny(seg_clean, 100, 200)
    colored_mask[edges > 0] = [0, 0, 255]  # Red edges
    overlay = cv2.addWeighted(img_np, 1-alpha, colored_mask, alpha, 0)
    return seg_clean, overlay

# # DeepLabV3+ segmentation (optional, if TF Hub is available)
# def deeplabv3_segment(img, alpha=0.4):
#     # Load model (only once)
#     if not hasattr(deeplabv3_segment, 'model'):
#         deeplabv3_segment.model = hub.load('https://tfhub.dev/tensorflow/deeplabv3-mobilenetv2/1')
    
#     img_np = np.array(img)
#     h, w = img_np.shape[:2]
    
#     # Preprocess & predict
#     resized = cv2.resize(img_np, (513, 513))
#     input_tensor = tf.convert_to_tensor(resized[None, ...], dtype=tf.uint8)
#     result = deeplabv3_segment.model(input_tensor)
#     mask = result['default'][0].numpy().astype(np.uint8)
#     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
#     # Apply a visually distinct colormap
#     colormap = np.array([
#         [0, 0, 0],       # Background (black)
#         [128, 0, 0],     # Class 1 (dark red)
#         [0, 128, 0],     # Class 2 (dark green)
#         [128, 128, 0],   # Class 3 (olive)
#         [0, 0, 128],     # Class 4 (dark blue)
#         [128, 0, 128],   # Class 5 (purple)
#         [0, 128, 128],   # Class 6 (teal)
#         [128, 128, 128], # Class 7 (gray)
#         [64, 0, 0],      # Class 8 (maroon)
#         [192, 0, 0],     # Class 9 (red)
#         [64, 128, 0],    # Class 10 (green-brown)
#         [192, 128, 0],   # Class 11 (orange)
#     ], dtype=np.uint8)
    
#     mask_color = colormap[mask % len(colormap)]
#     overlay = cv2.addWeighted(img_np, 1 - alpha, mask_color, alpha, 0)
    
#     # Highlight edges (contours) in red
#     for class_id in np.unique(mask):
#         if class_id == 0:  # Skip background
#             continue
#         class_mask = (mask == class_id).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Red edges
    
#     return mask, overlay 

def kmeans_segmentation(img, k=4):
    """
    Segment the image using K-means clustering in color space.
    Args:
        img: PIL Image
        k: Number of color clusters
    Returns:
        segmented_image: np.ndarray (segmented image)
    """
    img_np = np.array(img)
    Z = img_np.reshape((-1, 3))
    Z = np.float32(Z)
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((img_np.shape))
    return segmented_image 