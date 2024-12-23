'''
python modules/detect_object_phone_slot/script_batch_predict.py \
    --model dataset/model_object_detect_phone_slot/models/v2/best.onnx \
    --imgs dataset/model_object_detect_phone_slot/data/unseen
'''
import os
import cv2.dnn
import argparse
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
parser.add_argument("--imgs", default=str(ASSETS / "bus.jpg"), help="Path to input imagew.")
parser.add_argument("--debug",default="dst",help="folder debug path")
args = parser.parse_args()
    
# Inits
onnx_model = args.model
input_images = args.imgs
debug_path = args.debug
CLASSES = ['phone','gripper'] # yaml_load(check_yaml("coco128.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
images_tail = ['png','jpg','jpeg']

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
  
    # Create debug path   
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # Load the ONNX model
    model = cv2.dnn.readNetFromONNX(onnx_model)
    
    images= os.listdir(input_images)
    
    for image_idx in range(len(images)):
        
        # check image
        current_image = images[image_idx]
        image_tail = current_image.split('.')[-1]
        if image_tail not in images_tail:
            print(current_image+" is not support")
            continue
        
        # get image path
        input_image = os.path.join(input_images, current_image)

        # Read the input image
        original_image: np.ndarray = cv2.imread(input_image)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        detections = [] # [{'class_id':, 'class_name':, 'confidence':, 'box':[], 'scale':},...]
        
        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            
            draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )
            
        cv2.imwrite(os.path.join(debug_path, current_image), original_image)
        print(f'{image_idx}/{len(images)} {input_image}')
    