import datetime
import json
import logging
import math
import os
import time
import traceback
from typing import Tuple

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

# Inits
CONF_THRESHOLD = 0.7
CLASSES = ['magnetic_dark', 'rectangle_hole']
images_tail = ['png','jpg','jpeg']

class YOLOv8_Triton:
    def __init__(self, model_path, labelmap_path, threshold) -> None:
        # init param
        self.log = logging.getLogger(__name__)
        self.model_path = model_path
        self.cls_names = self.load_labelmap(labelmap_path)
        self.conf_thres = threshold
        # model_name_on_triton
        self.model_name = os.path.basename(self.model_path).split('.')[0]
        self.model_name_on_triton = '{}'.format(self.model_name)
        self.triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        print("cls_names = {}".format(self.cls_names))
    
    def _load_model_ready(self):
        is_model_ready = False
        try:
            if self.triton_client.is_server_live():
                logging.info("triton server is ready")
                if self.triton_client.is_model_ready(model_name=self.model_name_on_triton) is False:
                    logging.warning("triton model is not ready. Will load model {}".format(self.model_name_on_triton))
                    self.triton_client.load_model(model_name=self.model_name_on_triton)
                    
                for retry in range(10):
                    if self.triton_client.is_model_ready(model_name=self.model_name_on_triton) is False:
                        logging.warning("triton model is not ready; waiting in {} seconds".format(retry))
                        time.sleep(1)
                    else:
                        logging.info("triton model {} is ready; ready after {} seconds".format(self.model_name_on_triton, retry))
                        is_model_ready = True
                        break
        except Exception as e:
            logging.exception("Exception while")
        return is_model_ready
    
    def load_model(self): 
        pass

    def load_labelmap(self, labelmap_path):
        labels = {}
        with open(labelmap_path, "r") as file:
            for line in file:
                if 'id:' in line:
                    id = int(line.strip().replace('id:', '').strip())
                elif 'name:' in line:
                    name = line.strip().replace('name:', '').strip().strip("'")
                    labels[id] = name
        return labels
    
    def draw_area_box(self, image, area_in_config_file, color_bgr, size=0.005, line_type=cv2.LINE_AA, is_copy=True):
        x_from, y_from, x_to, y_to = area_in_config_file
        assert size > 0
        image = image.copy() if is_copy else image # copy/clone a new image
        # calculate thickness
        h, w = image.shape[:2]
        if size > 0:        
            short_edge = min(h, w)
            thickness = int(short_edge * size)
            thickness = 1 if thickness <= 0 else thickness
        else:
            thickness = -1
        # calc x,y in absolute coord
        cv2.rectangle(img=image, pt1=(x_from, y_from), pt2=(x_to, y_to), color=color_bgr, thickness=thickness, lineType=line_type, shift=0)
        #location = (x,y,w,h)
        location = (x_from, y_from, x_to - x_from, y_to - y_from)
        return image, location

class CalibMagnetArea(YOLOv8_Triton):
    def __init__(self, model_path, labelmap_path, threshold) -> None:
        YOLOv8_Triton.__init__(self, model_path, labelmap_path, threshold)
    
    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ':' + msg
        # Log message
        text_log_file.write(msg)
    
    def load_coordinates_to_crop(self, position):
        with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
            data = json.load(json_file)
        if position == 'left':
            x_from_1 = data.get('x_from_wall_left_1', 76) 
            x_to_1 = data.get('x_to_wall_left_1', 396)
            y_from_1 = data.get('y_from_wall_left_1', 260)
            y_to_1 = data.get('y_to_wall_left_1', 420)
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_wall_left_2', 800) 
            x_to_2 = data.get('x_to_wall_left_2', 1120)
            y_from_2 = data.get('y_from_wall_left_2', 260)
            y_to_2 = data.get('y_to_wall_left_2', 420)
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]

        elif position == 'right':
            x_from_1 = data.get('x_from_wall_right_1', 61)
            x_to_1 = data.get('x_to_wall_right_1', 381)
            y_from_1 = data.get('y_from_wall_right_1', 207)
            y_to_1 = data.get('y_to_wall_right_1', 367)
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_wall_right_2', 767) 
            x_to_2 = data.get('x_to_wall_right_2', 1087)
            y_from_2 = data.get('y_from_wall_right_2', 191)
            y_to_2 = data.get('y_to_wall_right_2', 351)
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]
            
        elif position == 'buffer':
            x_from_1 = data.get('x_from_kangaroo_1',100)
            x_to_1 = data.get('x_to_kangaroo_1',420)
            y_from_1 = data.get('y_from_kangaroo_1',420)
            y_to_1 = data.get('y_to_kangaroo_1',580)
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_kangaroo_2',800)
            x_to_2 = data.get('x_to_kangaroo_2',1120)
            y_from_2 = data.get('y_from_kangaroo_2',415)
            y_to_2 = data.get('y_to_kangaroo_2',575)
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]
            
        return position_crop_magnet_left,position_crop_magnet_right
              

    def crop_img(self, image_path, position_crop_magnet_left, position_crop_magnet_right):
        #image_crop = image[10:840, 90:1130, :]
        x_from_left, x_to_left, y_from_left, y_to_left = position_crop_magnet_left
        x_from_right, x_to_right, y_from_right, y_to_right = position_crop_magnet_right
        image = cv2.imread(image_path)
        #image_crop_left = image[x_from_left:x_to_left, y_from_left:y_to_left, :]
        image_crop_left = image[y_from_left:y_to_left, x_from_left:x_to_left, :]
        #image_crop_right = image[x_from_right:x_to_right, y_from_right:y_to_right, :]
        image_crop_right = image[y_from_right:y_to_right, x_from_right:x_to_right, :]
        return image_crop_left, image_crop_right

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
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
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    
    def letterbox(self, img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
   
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
    
    def preprocess_image(self, img0: np.ndarray):
        """
        Preprocess image according to YOLOv8 input requirements.
        Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

        Parameters:
        img0 (np.ndarray): image for preprocessing
        Returns:
        img (np.ndarray): image after preprocessing
        """
        # resize
        img = self.letterbox(img0)[0]

        # Convert HWC to CHW
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        return img
    
    def image_to_tensor(self, image:np.ndarray):
        """
        Preprocess image according to YOLOv8 input requirements.
        Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

        Parameters:
        img (np.ndarray): image for preprocessing
        Returns:
        input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
        """
        input_tensor = image.astype(np.float32)  # uint8 to fp32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        # add batch dimension
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor
    
    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
        y = np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y

    def box_iou(self, box1, box2, eps=1e-7):
        """
        Calculate intersection-over-union (IoU) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Args:
            box1 (numpy array): A numpy array of shape (N, 4) representing N bounding boxes.
            box2 (numpy array): A numpy array of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (numpy array): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
        """

        (a1, a2), (b1, b2) = np.split(np.expand_dims(box1, axis=1), 2), np.split(np.expand_dims(box2, axis=1), 2)
        inter = np.clip((np.min(a2, b2) - np.max(a1, b1)), a_min = 0, a_max = None) * 2

        # IoU = inter / (area1 + area2 - inter)
        return inter / (((a2 - a1) * 2)  + ((b2 - b1) * 2) - inter + eps)

    def nms(self, boxes, overlap_threshold=0.5, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        index_array = scores.argsort()[::-1]
        keep = []
        while index_array.size > 0:
            keep.append(index_array[0])
            x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
            y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
            x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
            y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

            w = np.maximum(0.0, x2_ - x1_ + 1)
            h = np.maximum(0.0, y2_ - y1_ + 1)
            inter = w * h

            if min_mode:
                overlap = inter / np.minimum(areas[index_array[0]], areas[index_array[1:]])
            else:
                overlap = inter / (areas[index_array[0]] + areas[index_array[1:]] - inter)

            inds = np.where(overlap <= overlap_threshold)[0]
            index_array = index_array[inds + 1]
        return keep
    
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                calculated based on the size difference between the two images.
            padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                rescaling.

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            if gain > 1:
                gain = 1.0
            pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
                (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., [0, 2]] -= pad[0]  # x padding
            boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes
    
    def clip_boxes(self, boxes, shape):
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def non_max_suppression(self,
            prediction,
            conf_thres=0.7,
            iou_thres=0.7,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nc=0,  # number of classes (optional)
            max_time_img=0.05,
            max_nms=30000,
            max_wh=7680,
    ):   
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (numpy array): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, numpy array]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[numpy array]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        output = []
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = np.max(prediction[:, 4:mi], axis=1)  > conf_thres  # candidates

        # Settings
        min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        prediction = np.transpose(prediction, (0, 2, 1))
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            #x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
        
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x=np.concatenate((x,v),axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            if nm > 0:
                box, cls, mask = np.split(x,(4, nc, nm),axis=1)
            else:
                box, cls = np.array_split(x,[4],axis=1)
                mask = np.zeros((box.shape[0],0))

            if multi_label:
                i, j = np.where(cls > conf_thres)
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None], mask[i]), 1)
            else:  # best class only
                conf= np.max(cls,axis=1)
                conf = conf.reshape(conf.shape[0],1)
                j =np.argmax(cls[:,:], axis=1, keepdims=True)
                x = np.concatenate((box, conf, j, mask), axis = 1)

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            scores = scores.reshape(scores.shape[0],1)
            con = np.concatenate((boxes,scores),axis=1)
            keep_boxes = self.nms(con, iou_thres)  # NMS 

            keep_boxes = keep_boxes[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[keep_boxes], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    keep_boxes = keep_boxes[iou.sum(1) > 1]  # require redundancy
            for k in keep_boxes:
                output.append(x[k])
            # if (time.time() - t) > time_limit:
            #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded
        return output

    def detect(self, image):
        
        # Write the text to the file
        self.log.info('In function detect!')
        
        self.log.info('model_name_on_triton: {}'.format(self.model_name_on_triton))
        is_model_ready = self._load_model_ready()
        if is_model_ready is False:
            self.log.warning("triton model {} is not ready; exit".format(self.model_name_on_triton))
            return [], None
        # Preprocessing image
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_image = self.preprocess_image(image)
        input_tensor = self.image_to_tensor(preprocessed_image)
        # Create input, output of request triton
        inputs = [grpcclient.InferInput('images', input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)
        outputs = [grpcclient.InferRequestedOutput('output0')]
        # Call to triton serving
        result = self.triton_client.infer(model_name=self.model_name_on_triton, inputs=inputs, outputs=outputs)
        boxes = []
        boxes = np.array(result.as_numpy('output0'))
        
        detections = self.non_max_suppression(boxes, self.conf_thres)
        
        # Postprocess for bbox
        detections = self.non_max_suppression(boxes, self.conf_thres)
        detections = np.array(detections)
        print("detection shape: ", len(detections))
        
        if len(detections) == 0:
            self.log.info("Object empty!")
            return None, None
        
        detections[:, :4] = self.scale_boxes((640, 640), detections[:, :4], (h, w)).round()
        
        bboxes = []
        object_list = []
        for bbox in detections:
            current_object = {}
            class_name = self.cls_names.get(int(bbox[5]))
            score = round(float(bbox[4]), 2)
            bboxes.append([class_name, score, int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])])
        
            current_object['label'] = self.cls_names.get(int(bbox[5]))
            current_object['conf'] = round(float(bbox[4]), 2)
            current_object['x'] = int(bbox[0])
            current_object['y'] = int(bbox[1])
            current_object['w'] = int(bbox[2])
            current_object['h'] = int(bbox[3])
            if (current_object['conf'] > self.conf_thres) and (current_object['label'] == 'magnetic_dark'):
                object_list.append(current_object)

        return object_list
    
    # def post_process(self, object_list_1, object_list_2, debug_img_1, debug_img_2, text_log_file):
        
    #     # Write the text to the file
    #     msg = 'In function post-proceess!\n'
    #     self.log_msg(msg, text_log_file)
    #     return result, image_pad
    
    def run(self, image_path_1, image_path_2, position_to_calib, log_file_path, ml_debug_path):
        # init logger
        text_log_file = open(log_file_path, 'w')
        result_dict, std_image_debug, cur_image_debug = {}, None, None
        try:
            # Crop images
            position_crop_magnet_left, position_crop_magnet_right = self.load_coordinates_to_crop(position_to_calib)
            print(type(position_crop_magnet_left))
            print(position_crop_magnet_left)
            print(position_crop_magnet_right)
            image_1_left, image_1_right = self.crop_img(image_path_1, position_crop_magnet_left, position_crop_magnet_right)
            image_2_left, image_2_right = self.crop_img(image_path_2, position_crop_magnet_left, position_crop_magnet_right)
            self.log_msg('End run function crop!\n', text_log_file)
            
            std_image_debug = cv2.imread(image_path_1)
            cur_image_debug = cv2.imread(image_path_2)
            cv2.rectangle(std_image_debug, (position_crop_magnet_left[0], position_crop_magnet_left[2]), (position_crop_magnet_left[1], position_crop_magnet_left[3]), (0, 255, 0), 2)
            cv2.rectangle(std_image_debug, (position_crop_magnet_right[0], position_crop_magnet_right[2]), (position_crop_magnet_right[1], position_crop_magnet_right[3]), (0, 255, 0), 2)
            cv2.rectangle(cur_image_debug, (position_crop_magnet_left[0], position_crop_magnet_left[2]), (position_crop_magnet_left[1], position_crop_magnet_left[3]), (0, 255, 0), 2)
            cv2.rectangle(cur_image_debug, (position_crop_magnet_right[0], position_crop_magnet_right[2]), (position_crop_magnet_right[1], position_crop_magnet_right[3]), (0, 255, 0), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_0.jpg')
            cv2.imwrite(path_image_debug, std_image_debug)
            path_image_debug = os.path.join(ml_debug_path, 'cur_image_debug_0.jpg')
            cv2.imwrite(path_image_debug, cur_image_debug)
            
            # detect
            self.log_msg('Start run function detect!\n', text_log_file)
            object_list_1_left, debug_img_1_left = self.detect(image_1_left, text_log_file)
            object_list_1_right, debug_img_1_right = self.detect(image_1_right, text_log_file)
            
            object_list_2_left, debug_img_2_left = self.detect(image_2_left, text_log_file)
            object_list_2_right, debug_img_2_right = self.detect(image_2_right, text_log_file)
            
            std_box_left_magnetic = dict()
            x_max = 0
            for box in object_list_1_left:
                x = box['x']
                if x > x_max:
                    x_max = x
                    std_box_left_magnetic = box
                    
            std_box_right_magnetic = dict()
            x_min = image_1_right.shape[1]
            for box in object_list_1_right:
                x = box['x']
                if x < x_min:
                    x_min = x
                    std_box_right_magnetic = box
            
            cur_box_left_magnetic = dict()
            x_max = 0
            for box in object_list_2_left:
                x = box['x']
                if x > x_max:
                    x_max = x
                    cur_box_left_magnetic = box
            
            cur_box_right_magnetic = dict()
            x_min = image_2_right.shape[1]
            for box in object_list_2_right:
                x = box['x']
                if x < x_min:
                    x_min = x
                    cur_box_right_magnetic = box
            
            std_image_debug_crop_left = std_image_debug[position_crop_magnet_left[2]:position_crop_magnet_left[3], position_crop_magnet_left[0]:position_crop_magnet_left[1]]
            cv2.rectangle(std_image_debug_crop_left, (std_box_left_magnetic['x'], std_box_left_magnetic['y']), (std_box_left_magnetic['x'] + std_box_left_magnetic['w'], std_box_left_magnetic['y'] + std_box_left_magnetic['h']), (255, 0, 0), 2)
            std_image_debug_crop_right = std_image_debug[position_crop_magnet_right[2]:position_crop_magnet_right[3], position_crop_magnet_right[0]:position_crop_magnet_right[1]]
            cv2.rectangle(std_image_debug_crop_right, (std_box_right_magnetic['x'], std_box_right_magnetic['y']), (std_box_right_magnetic['x'] + std_box_right_magnetic['w'], std_box_right_magnetic['y'] + std_box_right_magnetic['h']), (0, 255, 0), 2)
            
            cur_image_debug_crop_left = cur_image_debug[position_crop_magnet_left[2]:position_crop_magnet_left[3], position_crop_magnet_left[0]:position_crop_magnet_left[1]]
            cv2.rectangle(cur_image_debug_crop_left, (cur_box_left_magnetic['x'], cur_box_left_magnetic['y']), (cur_box_left_magnetic['x'] + cur_box_left_magnetic['w'], cur_box_left_magnetic['y'] + cur_box_left_magnetic['h']), (255, 0, 0), 2)
            cur_image_debug_crop_right = cur_image_debug[position_crop_magnet_right[2]:position_crop_magnet_right[3], position_crop_magnet_right[0]:position_crop_magnet_right[1]]
            cv2.rectangle(cur_image_debug_crop_right, (cur_box_right_magnetic['x'], cur_box_right_magnetic['y']), (cur_box_right_magnetic['x'] + cur_box_right_magnetic['w'], cur_box_right_magnetic['y'] + cur_box_right_magnetic['h']), (0, 255, 0), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_crop_left_1.jpg')
            cv2.imwrite(path_image_debug, std_image_debug_crop_left)
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_crop_right_1.jpg')
            cv2.imwrite(path_image_debug, std_image_debug_crop_right)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_1.jpg')
            cv2.imwrite(path_image_debug, std_image_debug)
            path_image_debug = os.path.join(ml_debug_path, 'cur_image_debug_1.jpg')
            cv2.imwrite(path_image_debug, cur_image_debug)
            
            std_mm_per_pixel_ratio = math.fabs(113/((position_crop_magnet_right[0] + std_box_right_magnetic['x']) - (position_crop_magnet_left[0] + std_box_left_magnetic['x'])))
            cur_mm_per_pixel_ratio = math.fabs(113/((position_crop_magnet_right[0] + cur_box_right_magnetic['x']) - (position_crop_magnet_left[0] + cur_box_left_magnetic['x'])))
            std_mm_per_pixel_ratio_rounded = np.round(std_mm_per_pixel_ratio, 6)
            cur_mm_per_pixel_ratio_rounded = np.round(cur_mm_per_pixel_ratio, 6)
            
            y_putText = int(cur_image_debug.shape[0]/3)
            x_putText = int(cur_image_debug.shape[1]/4)
            
            std_left_magnetic_center_x_in_pixel = int(std_box_left_magnetic['x'] + std_box_left_magnetic['w']/2)
            std_left_magnetic_center_y_in_pixel = int(std_box_left_magnetic['y'] + std_box_left_magnetic['h']/2)
            std_right_magnetic_center_x_in_pixel = int(std_box_right_magnetic['x'] + std_box_right_magnetic['w']/2)
            std_right_magnetic_center_y_in_pixel = int(std_box_right_magnetic['y'] + std_box_right_magnetic['h']/2)
            
            std_center_left = (std_left_magnetic_center_x_in_pixel + position_crop_magnet_left[0] , std_left_magnetic_center_y_in_pixel + position_crop_magnet_left[2])
            std_center_right = (std_right_magnetic_center_x_in_pixel + position_crop_magnet_right[0] , std_right_magnetic_center_y_in_pixel + position_crop_magnet_right[2])
            
            cv2.circle(std_image_debug, std_center_left, 5, (0, 0, 255), -1)
            cv2.circle(std_image_debug, std_center_right, 5, (0, 0, 255), -1)
            
            std_left_magnetic_center_x_in_mm = (std_box_left_magnetic['x'] + std_box_left_magnetic['w']/2) * std_mm_per_pixel_ratio
            std_left_magnetic_center_y_in_mm = (std_box_left_magnetic['y'] + std_box_left_magnetic['h']/2) * std_mm_per_pixel_ratio
            
            cur_left_magnetic_center_x_in_pixel = int(cur_box_left_magnetic['x'] + cur_box_left_magnetic['w']/2)
            cur_left_magnetic_center_y_in_pixel = int(cur_box_left_magnetic['y'] + cur_box_left_magnetic['h']/2)
            cur_right_magnetic_center_x_in_pixel = int(cur_box_right_magnetic['x'] + cur_box_right_magnetic['w']/2)
            cur_right_magnetic_center_y_in_pixel = int(cur_box_right_magnetic['y'] + cur_box_right_magnetic['h']/2)

            cur_center_left = (cur_left_magnetic_center_x_in_pixel + position_crop_magnet_left[0], cur_left_magnetic_center_y_in_pixel + position_crop_magnet_left[2])
            cur_center_right = (cur_right_magnetic_center_x_in_pixel + position_crop_magnet_right[0], cur_right_magnetic_center_y_in_pixel + position_crop_magnet_right[2])

            cv2.circle(cur_image_debug, cur_center_left, 5, (0, 0, 255), -1)
            cv2.circle(cur_image_debug, cur_center_right, 5, (0, 0, 255), -1)

            std_center = (round((std_center_left[0] + std_center_right[0])/2) , round((std_center_left[1] + std_center_right[1])/2))
            cur_center = (round((cur_center_left[0] + cur_center_right[0])/2) , round((cur_center_left[1] + cur_center_right[1])/2))
            cv2.circle(cur_image_debug, cur_center, 5, (0, 0, 255), -1)
            cv2.circle(std_image_debug, std_center, 5, (0, 0, 255), -1)

            cv2.putText(std_image_debug, "x: {}, y: {}, r: {}".format(std_center[0], std_center[1], std_mm_per_pixel_ratio_rounded), (x_putText, y_putText), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(cur_image_debug, "x: {}, y: {}, r: {}".format(cur_center[0], cur_center[1], cur_mm_per_pixel_ratio_rounded), (x_putText, y_putText), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cur_left_magnetic_center_x_in_mm = (cur_box_left_magnetic['x'] + cur_box_left_magnetic['w']/2) * cur_mm_per_pixel_ratio
            cur_left_magnetic_center_y_in_mm = (cur_box_left_magnetic['y'] + cur_box_left_magnetic['h']/2) * cur_mm_per_pixel_ratio

            std_center_x_in_mm = np.round(std_center[0] * std_mm_per_pixel_ratio, 6)
            std_center_y_in_mm = np.round(std_center[1] * std_mm_per_pixel_ratio, 6)
            cur_center_x_in_mm = np.round(cur_center[0] * cur_mm_per_pixel_ratio, 6)
            cur_center_y_in_mm = np.round(cur_center[1] * cur_mm_per_pixel_ratio, 6)

            delta_center_x_in_mm = np.round(cur_center_x_in_mm - std_center_x_in_mm, 4)
            delta_center_y_in_mm = -np.round(cur_center_y_in_mm - std_center_y_in_mm, 4)

            # delta_center_x_in_mm = np.round(cur_left_magnetic_center_x_in_mm - std_left_magnetic_center_x_in_mm, 6)
            # delta_center_y_in_mm = -np.round(cur_left_magnetic_center_y_in_mm - std_left_magnetic_center_y_in_mm, 6)
            
            focal_length_in_mm = 8
            pixel_size_in_mm = 0.0045
            magnetic_distance_in_mm = 113
            
            cur_magnetic_distance_in_pixel = math.fabs(cur_center_left[0] - cur_center_right[0])
            std_magnetic_distance_in_pixel = math.fabs(std_center_left[0] - std_center_right[0])
            
            cur_hypotenuse_distance_between_camera_and_magnetic = (focal_length_in_mm / (pixel_size_in_mm * cur_magnetic_distance_in_pixel)) * magnetic_distance_in_mm
            std_hypotenuse_distance_between_camera_and_magnetic = (focal_length_in_mm / (pixel_size_in_mm * std_magnetic_distance_in_pixel)) * magnetic_distance_in_mm
            
            cur_line_distance_between_camera_and_magnetic = round(cur_hypotenuse_distance_between_camera_and_magnetic * math.cos(40))
            std_line_distance_between_camera_and_magnetic = round(std_hypotenuse_distance_between_camera_and_magnetic * math.cos(40))
            
            delta_center_l_in_mm = np.round(cur_line_distance_between_camera_and_magnetic - std_line_distance_between_camera_and_magnetic, 4)
            
            if position_to_calib == 'right':
                delta_center_x_in_mm = -delta_center_x_in_mm
                delta_center_y_in_mm = delta_center_y_in_mm
                cv2.putText(cur_image_debug, "Right: offset_x: {}, offset_z: {}, offset_l: {}".format(delta_center_x_in_mm, delta_center_y_in_mm, delta_center_l_in_mm), (x_putText, y_putText + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif position_to_calib == 'left':
                cv2.putText(cur_image_debug, "Left: offset_x: {}, offset_z: {}, offset_l: {}".format(delta_center_x_in_mm, delta_center_y_in_mm, delta_center_l_in_mm), (x_putText, y_putText + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif position_to_calib == 'buffer':
                cv2.putText(cur_image_debug, "Buffer: offset_x: {}, offset_z: {}, offset_l: {}".format(delta_center_x_in_mm, delta_center_y_in_mm, delta_center_l_in_mm), (x_putText, y_putText + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'std_image_debug_2.jpg')
            cv2.imwrite(path_image_debug, std_image_debug)
            path_image_debug = os.path.join(ml_debug_path, 'cur_image_debug_2.jpg')
            cv2.imwrite(path_image_debug, cur_image_debug)

            # draw debug
            result_dict = dict()
            # result_dict["offset_x"] = offset_x
            # result_dict["offset_y"] = offset_y
            result_dict["offset_x"] = delta_center_x_in_mm
            result_dict["offset_y"] = delta_center_y_in_mm
            print("result_dict", result_dict)
            self.log_msg('Draw debug image successful\n', text_log_file)
            
        except Exception as e:
            self.log_msg('Error: ' + str(e) + '\n', text_log_file)
            #write traceback to logfile
            traceback.print_exc()
            traceback.print_exc(file=text_log_file)
            
            text_log_file.close()
            return result_dict, std_image_debug, cur_image_debug
        
        # Close file
        text_log_file.close()
        return result_dict, std_image_debug, cur_image_debug

if __name__ == '__main__':
    position_to_calib = 'left'
    image_path_1 = "/home/greystone/StorageWall/image_debug/calibrate/image_org/original_phone_slot_right.png"
    image_path_2 = "/home/greystone/StorageWall/image_debug/calibrate/image_org/calibration_W01RSR39C04_6.png"
    path_log_id_phone_slot = "/home/greystone/StorageWall/image_debug/calibrate/calibrate_debug/log_id_phone_slot.txt"
    model_weight_path = '/home/greystone/StorageWall/model_template/Calibrate/calibrate.onnx'
    ml_debug_path = "/home/greystone/StorageWall/image_debug/calibrate/calibrate_debug"
    object_detector = CalibMagnetArea(model_weight_path, 0.70)
    result, image_pad_debug_1, image_pad_debug_2 = object_detector.run(image_path_1, image_path_2, position_to_calib, path_log_id_phone_slot, ml_debug_path)
