import json
import cv2
import numpy as np
from PIL import Image


json_path = 'E:/mask/Image 1_annotation_ens.json'
image_path = 'E:/img/P6290001.JPG'  # Your original image
output_mask_path = 'E:/output_mask/P6290038.png'
output_annotated_path = 'E:/output_annotation/P6290038.jpg'

with open(json_path, 'r') as f:
    data = json.load(f)

image = cv2.imread(image_path)
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Single channel binary mask

for obj in data['annotation']['objects']:
    points = np.array(obj['points']['exterior'], dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)  # Binary mask
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)  # Green annotation

cv2.imwrite(output_mask_path, mask)
cv2.imwrite(output_annotated_path, image)