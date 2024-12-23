import os
import cv2
import time
import math
import numpy as np
import datetime
import onnxruntime

class SegmentPlasticBox:
    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        # Initialize model
        self.initialize_model(model_path)
        self.model_path = model_path
        print('[PlasticBoxDetector] Load model successful!')
        
        self.W = 1200
        self.H = 600
        self.match_color = (0,255,100)
        self.PLASTIC_BOX_MIN_DIM = 5 # minimum dimension of phone object
        self.thickness = 5
        self.result_have_plastic_box = 'have_plastic_box'
        self.result_none_plastic_box = 'none_plastic_box'
        self.tag_log = '[PlasticBoxDetector]'
        self.class_names = ['plastic_box']
        self.rng = np.random.default_rng(3)
        self.colors = self.rng.uniform(0, 255, size=(len(self.class_names), 3))
        
    def __call__(self, image):
        return self.segment_objects(image)
    
    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()
    
    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return self.draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return self.draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes
    
    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ': ' + msg
        # Log message
        text_log_file.write(msg)

    # Create a list of colors for each class where each color is a tuple of 3 integer values
    

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes


    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou


    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = self.draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return mask_img


    def draw_masks(self, image, object_list, mask_alpha=0.3, mask_maps=None):
        mask_img = image.copy()
        boxes, scores, class_ids, mask_maps = object_list
        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill mask image
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


    def draw_comparison(self, img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
        (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=fontsize, thickness=text_thickness)
        x1 = img1.shape[1] // 3
        y1 = th
        offset = th // 5
        cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                    (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
        cv2.putText(img1, name1,
                    (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, fontsize,
                    (255, 255, 255), text_thickness)

        (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=fontsize, thickness=text_thickness)
        x1 = img2.shape[1] // 3
        y1 = th
        offset = th // 5
        cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                    (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

        cv2.putText(img2, name2,
                    (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, fontsize,
                    (255, 255, 255), text_thickness)

        combined_img = cv2.hconcat([img1, img2])
        if combined_img.shape[1] > 3840:
            combined_img = cv2.resize(combined_img, (3840, 2160))

        return combined_img


    def detect(self, image_path, text_log_file):
        
        # Write the text to the file        
        msg = 'In function detect!\n'
        self.log_msg(msg, text_log_file)
        
        # Init
        object_list = []
        
        # Predict
        image = cv2.imread(image_path)
        if image is None:
            msg = 'Error: Image not found!\n'
            self.log_msg(msg, text_log_file)
            return object_list, None
        
        boxes, scores, class_ids, mask_maps = self.segment_objects(image)
        object_list = boxes, scores, class_ids, mask_maps
        
        return object_list, image
        
    def post_process(self, object_list, debug_img, text_log_file):
        
        # Write the text to the file
        msg = 'In function post-process!\n'
        self.log_msg(msg, text_log_file)
        
        # init
        result = self.result_none_plastic_box # 'have_plastic_box', 'none_plastic_box'
        
        # check object detected
        if len(object_list) == 0:
            return result + '_object_list_empty', debug_img
        else:
            combined_img = self.draw_masks(debug_img, object_list)
            result = self.result_have_plastic_box

        return result, combined_img
    
    def run(self, image_path, log_file_path):
        # init logger
        text_log_file = open(log_file_path, 'w')
        try:
            # detect
            self.log_msg('Start run function detect!\n', text_log_file)
            object_list, debug_img = self.detect(image_path, text_log_file)
            #image = cv2.imread(image_path)
            #objects_list = plastic_box_detector.segment_objects(image)
            #boxes, scores, class_ids, mask_maps = objects_list
            self.log_msg('End run function detect!\n', text_log_file)
            
            # Print out current object
            msg  = 'Current object list: ' + str(object_list) + '\n'
            self.log_msg(msg, text_log_file)
            
            # post process
            self.log_msg('Start run function post-process!\n', text_log_file)
            result, debug_img = self.post_process(object_list, debug_img, text_log_file)
            self.log_msg('End run function post-process!\n', text_log_file)
            
            msg  = 'Result=' + result + '\n'
            self.log_msg(msg, text_log_file)
            
            # draw debug
            cv2.putText(debug_img, result, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
            self.log_msg('Draw debug image successful\n', text_log_file)
            
            # Close file
            text_log_file.close()
            
            return result, debug_img
        
        except Exception as e:
            self.log_msg('Error: ' + str(e) + '\n', text_log_file)
            text_log_file.close()
            return self.result_none_plastic_box, None
        
if __name__ == '__main__':   
    image_path = "/home/greystone/StorageWall/image_debug/1712905725249_172_detect_phone_slot_org.jpg"
    path_log_id_phone_slot = "/home/greystone/StorageWall/image_debug/log_id_phone_slot.txt"
    model_phone_slot_weight_path = '/home/greystone/StorageWall/model_template/PlasticBoxs/model_segment_plastic_box.onnx'
    ml_debug_path = "/home/greystone/StorageWall/image_debug"
    plastic_box_detector = SegmentPlasticBox(model_phone_slot_weight_path, conf_thres=0.3, iou_thres=0.5)
    
    result, debug_img = plastic_box_detector.run(image_path, path_log_id_phone_slot)
    
        # Detect Objects
    # image = cv2.imread(image_path)
    # boxes, scores, class_ids, mask_maps = plastic_box_detector.segment_objects(image)

    # Draw detections
    # combined_img = plastic_box_detector.draw_masks(image, boxes, class_ids)
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", combined_img)
    # cv2.waitKey(0)
    
    cv2.imwrite(os.path.join(ml_debug_path,'debug_img_detect_phone.jpg'),debug_img)