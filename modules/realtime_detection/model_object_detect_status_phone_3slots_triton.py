import datetime
import json
import logging
import math
import os
import time
from typing import Tuple

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


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

class CalibMagnetArea_3slots(YOLOv8_Triton):
    def __init__(self, model_path, labelmap_path, threshold) -> None:
        YOLOv8_Triton.__init__(self, model_path, labelmap_path, threshold)

    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ':' + msg
        # Log message
        text_log_file.write(msg)
    
    def load_coordinates_to_crop(self, position, number_slots_detector):
        position_up_crop_magnet = None
        position_crop_magnet = None
        position_down_crop_magnet = None
        with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
            
            data = json.load(json_file)
        if number_slots_detector == 3:
            if position == 'left':
                x_up_from_1 = data.get('x_up_from_wall_left_1') 
                x_up_to_1 = data.get('x_up_to_wall_left_1')
                y_up_from_1 = data.get('y_up_from_wall_left_1')
                y_up_to_1 = data.get('y_up_to_wall_left_1')
                position_up_crop_magnet_left = [x_up_from_1, x_up_to_1, y_up_from_1, y_up_to_1]
                x_up_from_2 = data.get('x_up_from_wall_left_2') 
                x_up_to_2 = data.get('x_up_to_wall_left_2')
                y_up_from_2 = data.get('y_up_from_wall_left_2')
                y_up_to_2 = data.get('y_up_to_wall_left_2')
                position_up_crop_magnet_right = [x_up_from_2, x_up_to_2, y_up_from_2, y_up_to_2]
                position_up_crop_magnet = [position_up_crop_magnet_left, position_up_crop_magnet_right]
                
                x_down_from_1 = data.get('x_down_from_wall_left_1') 
                x_down_to_1 = data.get('x_down_to_wall_left_1')
                y_down_from_1 = data.get('y_down_from_wall_left_1')
                y_down_to_1 = data.get('y_down_to_wall_left_1')
                position_down_crop_magnet_left = [x_down_from_1, x_down_to_1, y_down_from_1, y_down_to_1]
                x_down_from_2 = data.get('x_down_from_wall_left_2') 
                x_down_to_2 = data.get('x_down_to_wall_left_2')
                y_down_from_2 = data.get('y_down_from_wall_left_2')
                y_down_to_2 = data.get('y_down_to_wall_left_2')
                position_down_crop_magnet_right = [x_down_from_2, x_down_to_2, y_down_from_2, y_down_to_2]
                position_down_crop_magnet = [position_down_crop_magnet_left, position_down_crop_magnet_right]

            elif position == 'right':
                x_up_from_1 = data.get('x_up_from_wall_right_1')
                x_up_to_1 = data.get('x_up_to_wall_right_1')
                y_up_from_1 = data.get('y_up_from_wall_right_1')
                y_up_to_1 = data.get('y_up_to_wall_right_1')
                position_up_crop_magnet_left = [x_up_from_1, x_up_to_1, y_up_from_1, y_up_to_1]
                x_up_from_2 = data.get('x_up_from_wall_right_2') 
                x_up_to_2 = data.get('x_up_to_wall_right_2')
                y_up_from_2 = data.get('y_up_from_wall_right_2')
                y_up_to_2 = data.get('y_up_to_wall_right_2')
                position_up_crop_magnet_right = [x_up_from_2, x_up_to_2, y_up_from_2, y_up_to_2]
                position_up_crop_magnet = [position_up_crop_magnet_left, position_up_crop_magnet_right]
                
                x_down_from_1 = data.get('x_down_from_wall_right_1')
                x_down_to_1 = data.get('x_down_to_wall_right_1')
                y_down_from_1 = data.get('y_down_from_wall_right_1')
                y_down_to_1 = data.get('y_down_to_wall_right_1')
                position_down_crop_magnet_left = [x_down_from_1, x_down_to_1, y_down_from_1, y_down_to_1]
                x_down_from_2 = data.get('x_down_from_wall_right_2') 
                x_down_to_2 = data.get('x_down_to_wall_right_2')
                y_down_from_2 = data.get('y_down_from_wall_right_2')
                y_down_to_2 = data.get('y_down_to_wall_right_2')
                position_down_crop_magnet_right = [x_down_from_2, x_down_to_2, y_down_from_2, y_down_to_2]
                position_down_crop_magnet = [position_down_crop_magnet_left, position_down_crop_magnet_right]
                
        if position == 'left':
            x_from_1 = data.get('x_from_wall_left_1') 
            x_to_1 = data.get('x_to_wall_left_1')
            y_from_1 = data.get('y_from_wall_left_1')
            y_to_1 = data.get('y_to_wall_left_1')
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_wall_left_2') 
            x_to_2 = data.get('x_to_wall_left_2')
            y_from_2 = data.get('y_from_wall_left_2')
            y_to_2 = data.get('y_to_wall_left_2')
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]
            position_crop_magnet = [position_crop_magnet_left, position_crop_magnet_right]
        elif position == 'right':
            x_from_1 = data.get('x_from_wall_right_1')
            x_to_1 = data.get('x_to_wall_right_1')
            y_from_1 = data.get('y_from_wall_right_1')
            y_to_1 = data.get('y_to_wall_right_1')
            position_crop_magnet_left = [x_from_1, x_to_1, y_from_1, y_to_1]
            x_from_2 = data.get('x_from_wall_right_2') 
            x_to_2 = data.get('x_to_wall_right_2')
            y_from_2 = data.get('y_from_wall_right_2')
            y_to_2 = data.get('y_to_wall_right_2')
            position_crop_magnet_right = [x_from_2, x_to_2, y_from_2, y_to_2]
            position_crop_magnet = [position_crop_magnet_left, position_crop_magnet_right]

        return position_up_crop_magnet, position_crop_magnet, position_down_crop_magnet
              
    def crop_img(self, image_org, position_crop_magnet):
        #image_crop = image[10:840, 90:1130, :]
        position_crop_magnet_left, position_crop_magnet_right = position_crop_magnet
        x_from_left, x_to_left, y_from_left, y_to_left = position_crop_magnet_left
        x_from_right, x_to_right, y_from_right, y_to_right = position_crop_magnet_right

        image = image_org
        #image_crop_left = image[x_from_left:x_to_left, y_from_left:y_to_left, :]
        image_crop_left = image[y_from_left:y_to_left, x_from_left:x_to_left, :]
        #image_crop_right = image[x_from_right:x_to_right, y_from_right:y_to_right, :]
        image_crop_right = image[y_from_right:y_to_right, x_from_right:x_to_right, :]
        return image_crop_left, image_crop_right
    
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
            return []
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
        
        # Postprocess for bbox
        detections = self.non_max_suppression(boxes, self.conf_thres)
        detections = np.array(detections)
        self.log.info('detection shape: {}'.format(len(detections)))
        
        if len(detections) == 0:
            self.log.info("Object empty!")
            return []
        
        detections[:, :4] = self.scale_boxes((640, 640), detections[:, :4], (h, w)).round()
        bboxes = []
        object_list = []
        for bbox in detections:
            current_object = {}
            class_name = self.cls_names.get(int(bbox[5]))
            score = round(float(bbox[4]), 2)
            bboxes.append([class_name, score, int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])])
        
            current_object['label'] = self.cls_names.get(int(bbox[5]) + 1)

            current_object['conf'] = round(float(bbox[4]), 2)
            current_object['x'] = int(bbox[0])
            current_object['y'] = int(bbox[1])
            current_object['w'] = int(bbox[2]) - int(bbox[0])
            current_object['h'] = int(bbox[3]) - int(bbox[1])
            if (current_object['conf'] > self.conf_thres) and (current_object['label'] == 'magnetic_dark'):
                object_list.append(current_object)

        return object_list
    
    # def post_process(self, object_list_1, object_list_2, debug_img_1, debug_img_2, text_log_file):
        
    #     # Write the text to the file
    #     msg = 'In function post-proceess!\n'
    #     self.log_msg(msg, text_log_file)
    #     return result, image_pad
    
    def run(self, image_org, image_debug, side_wall, ml_debug_path, number_slots_detector, position_have_phone):
        # init logger
        result_dict, std_image_debug, cur_image_debug = {}, None, None
        try:
            # Crop images
            position_crop_up, position_crop, position_crop_down = self.load_coordinates_to_crop(side_wall, number_slots_detector)
            
            print(position_crop_up)
            print(position_crop)
            print(position_crop_down)
            
            image_1_left, image_1_right = self.crop_img(image_org, position_crop_up)
            image_2_left, image_2_right = self.crop_img(image_org, position_crop)
            image_3_left, image_3_right = self.crop_img(image_org, position_crop_down)
            cv2.imwrite(os.path.join(ml_debug_path, "image_1_left.png"), image_1_left)
            cv2.imwrite(os.path.join(ml_debug_path, "image_1_right.png"), image_1_right)
            cv2.imwrite(os.path.join(ml_debug_path, "image_2_left.png"), image_2_left)
            cv2.imwrite(os.path.join(ml_debug_path, "image_2_right.png"), image_2_right)
            cv2.imwrite(os.path.join(ml_debug_path, "image_3_left.png"), image_3_left)
            cv2.imwrite(os.path.join(ml_debug_path, "image_3_right.png"), image_3_right)

            self.log.info('End run function crop!')
            
            cv2.rectangle(image_debug, (position_crop_up[0][0], position_crop_up[0][2]), (position_crop_up[0][1], position_crop_up[0][3]), (0, 255, 255), 2)
            cv2.rectangle(image_debug, (position_crop_up[1][0], position_crop_up[1][2]), (position_crop_up[1][1], position_crop_up[1][3]), (0, 255, 255), 2)
            
            cv2.rectangle(image_debug, (position_crop[0][0], position_crop[0][2]), (position_crop[0][1], position_crop[0][3]), (0, 255, 255), 2)
            cv2.rectangle(image_debug, (position_crop[1][0], position_crop[1][2]), (position_crop[1][1], position_crop[1][3]), (0, 255, 255), 2)
            
            cv2.rectangle(image_debug, (position_crop_down[0][0], position_crop_down[0][2]), (position_crop_down[0][1], position_crop_down[0][3]), (0, 255, 255), 2)
            cv2.rectangle(image_debug, (position_crop_down[1][0], position_crop_down[1][2]), (position_crop_down[1][1], position_crop_down[1][3]), (0, 255, 255), 2)
            
            path_image_debug = os.path.join(ml_debug_path, 'image_debug_1.jpg')
            cv2.imwrite(path_image_debug, image_debug)
            
            # detect
            self.log.info('Start run function detect!')
            flag_1 = 0
            object_list_1_left = self.detect(image_1_left)
            print("sai o day ne")
            self.log.info("object_list_1_left = {}".format(object_list_1_left) )
            object_list_1_right = self.detect(image_1_right)
            self.log.info("object_list_1_right = {}".format(object_list_1_right))
            if len(object_list_1_left) == 0 or len(object_list_1_right) == 0:
                flag_1 = 1
            self.log.info('object_list_1_left = {}, object_list_1_right = {}'.format(object_list_1_left, object_list_1_right))
            
            flag_2 = 0
            object_list_2_left = self.detect(image_2_left)
            self.log.info("object_list_2_left = {}".format(object_list_2_left))
            object_list_2_right = self.detect(image_2_right)
            self.log.info("object_list_2_right = {}".format(object_list_2_right))
            if len(object_list_2_left) == 0 or len(object_list_2_right) == 0:
                flag_2 = 1
            self.log.info('object_list_2_left = {}, object_list_2_right = {}'.format(object_list_2_left, object_list_2_right))
            
            flag_3 = 0
            object_list_3_left = self.detect(image_3_left)
            self.log.info("object_list_3_left = {}".format(object_list_3_left))
            object_list_3_right = self.detect(image_3_right)
            self.log.info("object_list_3_left = {}".format(object_list_3_left))
            if len(object_list_3_left) == 0 or len(object_list_3_right) == 0:
                flag_3 = 1
            self.log.info('object_list_3_left = {}, object_list_3_right = {}'.format(object_list_3_left, object_list_3_right))
            
            self.log.info('flag_1 = {}, flag_2 = {}, flag_3 = {}'.format(flag_1, flag_2, flag_3))
            
            if not flag_1:
                box_1_left_magnetic = dict()
                x_max = 0
                for box in object_list_1_left:
                    x = box['x']
                    if x >= x_max:
                        x_max = x
                        box_1_left_magnetic = box
                        
                box_1_right_magnetic = dict()
                x_min = image_1_right.shape[1]
                for box in object_list_1_right:
                    x = box['x']
                    if x <= x_min:
                        x_min = x
                        box_1_right_magnetic = box
                self.log.info("box_1_left_magnetic = ", box_1_left_magnetic)
                self.log.info("box_1_right_magnetic = ", box_1_right_magnetic)
                img_up_mm_per_pixel_ratio = math.fabs(116/((position_crop_up[1][0] + box_1_right_magnetic['x']) - (position_crop_up[0][0] + box_1_left_magnetic['x'])))
                img_up_mm_per_pixel_ratio_rounded = np.round(img_up_mm_per_pixel_ratio, 6)
                img_up_left_magnetic_center_x_in_pixel = int(box_1_left_magnetic['x'] + box_1_left_magnetic['w']/2)
                img_up_left_magnetic_center_y_in_pixel = int(box_1_left_magnetic['y'] + box_1_left_magnetic['h']/2)
                img_up_right_magnetic_center_x_in_pixel = int(box_1_right_magnetic['x'] + box_1_right_magnetic['w']/2)
                img_up_right_magnetic_center_y_in_pixel = int(box_1_left_magnetic['y'] + box_1_right_magnetic['h']/2)
                img_up_center_left = (img_up_left_magnetic_center_x_in_pixel + position_crop_up[0][0] , img_up_left_magnetic_center_y_in_pixel + position_crop_up[0][2])
                img_up_center_right = (img_up_right_magnetic_center_x_in_pixel + position_crop_up[1][0] , img_up_right_magnetic_center_y_in_pixel + position_crop_up[1][2])
                cv2.circle(image_debug, img_up_center_left, 5, (0, 0, 255), -1)
                cv2.circle(image_debug, img_up_center_right, 5, (0, 0, 255), -1)        
                
            if not flag_2:
                box_2_left_magnetic = dict()
                x_max = 0
                for box in object_list_2_left:
                    x = box['x']
                    if x >= x_max:
                        x_max = x
                        box_2_left_magnetic = box
                
                box_2_right_magnetic = dict()
                x_min = image_2_right.shape[1]
                for box in object_list_2_right:
                    x = box['x']
                    if x <= x_min:
                        x_min = x
                        box_2_right_magnetic = box
                        
                self.log.info("box_2_left_magnetic = ", box_2_left_magnetic)
                self.log.info("box_2_right_magnetic = ", box_2_right_magnetic)
                img_center_mm_per_pixel_ratio = math.fabs(116/((position_crop[1][0] + box_2_right_magnetic['x']) - (position_crop[0][0] + box_2_left_magnetic['x'])))
                img_center_mm_per_pixel_ratio_rounded = np.round(img_center_mm_per_pixel_ratio, 6)
                img_center_left_magnetic_center_x_in_pixel = int(box_2_left_magnetic['x'] + box_2_left_magnetic['w']/2)
                img_center_left_magnetic_center_y_in_pixel = int(box_2_left_magnetic['y'] + box_2_left_magnetic['h']/2)
                img_center_right_magnetic_center_x_in_pixel = int(box_2_right_magnetic['x'] + box_2_right_magnetic['w']/2)
                img_center_right_magnetic_center_y_in_pixel = int(box_2_left_magnetic['y'] + box_2_right_magnetic['h']/2)
                img_center_center_left = (img_center_left_magnetic_center_x_in_pixel + position_crop[0][0] , img_center_left_magnetic_center_y_in_pixel + position_crop[0][2])
                img_center_center_right = (img_center_right_magnetic_center_x_in_pixel + position_crop[1][0] , img_center_right_magnetic_center_y_in_pixel + position_crop[1][2])
                cv2.circle(image_debug, img_center_center_left, 5, (0, 0, 255), -1)
                cv2.circle(image_debug, img_center_center_right, 5, (0, 0, 255), -1)

            if not flag_3:
                box_3_left_magnetic = dict()
                x_max = 0
                for box in object_list_3_left:
                    x = box['x']
                    if x >= x_max:
                        x_max = x
                        box_3_left_magnetic = box
                
                box_3_right_magnetic = dict()
                x_min = image_2_right.shape[1]
                for box in object_list_3_right:
                    x = box['x']
                    if x <= x_min:
                        x_min = x
                        box_3_right_magnetic = box
                
                self.log.info("box_3_left_magnetic = ", box_3_left_magnetic)
                self.log.info("box_3_right_magnetic = ", box_3_right_magnetic)
                img_down_mm_per_pixel_ratio = math.fabs(116/((position_crop_down[1][0] + box_3_right_magnetic['x']) - (position_crop_down[0][0] + box_3_left_magnetic['x'])))
                img_down_mm_per_pixel_ratio_rounded = np.round(img_down_mm_per_pixel_ratio, 6)
                img_down_left_magnetic_center_x_in_pixel = int(box_3_left_magnetic['x'] + box_3_left_magnetic['w']/2)
                img_down_left_magnetic_center_y_in_pixel = int(box_3_left_magnetic['y'] + box_3_left_magnetic['h']/2)
                img_down_right_magnetic_center_x_in_pixel = int(box_3_right_magnetic['x'] + box_3_right_magnetic['w']/2)
                img_down_right_magnetic_center_y_in_pixel = int(box_3_left_magnetic['y'] + box_3_right_magnetic['h']/2)
                img_down_center_left = (img_down_left_magnetic_center_x_in_pixel + position_crop_down[0][0] , img_down_left_magnetic_center_y_in_pixel + position_crop_down[0][2])
                img_down_center_right = (img_down_right_magnetic_center_x_in_pixel + position_crop_down[1][0] , img_down_right_magnetic_center_y_in_pixel + position_crop_down[1][2])
                cv2.circle(image_debug, img_down_center_left, 5, (0, 0, 255), -1)
                cv2.circle(image_debug, img_down_center_right, 5, (0, 0, 255), -1)


            path_image_debug = os.path.join(ml_debug_path, 'image_debug_2.jpg')
            cv2.imwrite(path_image_debug, image_debug)

            position_phone_up, position_phone_center, position_phone_down = position_have_phone
            
            self.log.info('position_phone_up = {}, position_phone_center = {}, position_phone_down = {}'.format(position_phone_up, position_phone_center, position_phone_down))
            print('position_phone_up = {}, position_phone_center = {}, position_phone_down = {}'.format(position_phone_up, position_phone_center, position_phone_down))
            result_dict = dict()
            
            if (position_phone_up is not None) and not flag_1:
                delta_distance_phone_slot_up_left_in_pixel = position_phone_up[0] - img_up_center_left[0]
                delta_distance_phone_slot_up_right_in_pixel = img_up_center_right[0] - position_phone_up[2]
                delta_distance_phone_slot_up_left_in_mm = np.round(delta_distance_phone_slot_up_left_in_pixel * img_up_mm_per_pixel_ratio, 2)
                delta_distance_phone_slot_up_right_in_mm = np.round(delta_distance_phone_slot_up_right_in_pixel * img_up_mm_per_pixel_ratio, 2)
                cv2.putText(image_debug, str(delta_distance_phone_slot_up_left_in_mm) +"mm", (img_up_center_left[0] + 20, img_up_center_left[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)
                cv2.putText(image_debug, str(delta_distance_phone_slot_up_right_in_mm) +"mm", (img_up_center_right[0] - 60, img_up_center_right[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)
                if (delta_distance_phone_slot_up_left_in_mm > 5 and delta_distance_phone_slot_up_right_in_mm > 5):
                   result_dict["phone_1"] = "not_misaligned"
                else: result_dict["phone_1"] = "misaligned"
            else: result_dict["phone_1"] = "not_misaligned"
            
            if (position_phone_center is not None) and not flag_2:
                delta_distance_phone_slot_center_left_in_pixel = position_phone_center[0] - img_center_center_left[0]
                delta_distance_phone_slot_center_right_in_pixel = img_center_center_right[0] - position_phone_center[2]
                delta_distance_phone_slot_center_left_in_mm = np.round(delta_distance_phone_slot_center_left_in_pixel * img_center_mm_per_pixel_ratio, 2)
                delta_distance_phone_slot_center_right_in_mm = np.round(delta_distance_phone_slot_center_right_in_pixel * img_center_mm_per_pixel_ratio, 2)
                cv2.putText(image_debug, str(delta_distance_phone_slot_center_left_in_mm) +"mm", (img_center_center_left[0] + 20, img_center_center_left[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)
                cv2.putText(image_debug, str(delta_distance_phone_slot_center_right_in_mm) +"mm", (img_center_center_right[0] - 60, img_center_center_right[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)
                if (delta_distance_phone_slot_center_left_in_mm > 5 and delta_distance_phone_slot_center_right_in_mm > 5):
                   result_dict["phone_2"] = "not_misaligned"
                else: result_dict["phone_2"] = "misaligned"
            else: result_dict["phone_2"] = "not_misaligned"
            
            if (position_phone_down is not None) and not flag_3:
                delta_distance_phone_slot_down_left_in_pixel = position_phone_down[0] - img_down_center_left[0]
                delta_distance_phone_slot_down_right_in_pixel = img_down_center_right[0] - position_phone_down[2]
                delta_distance_phone_slot_down_left_in_mm = np.round(delta_distance_phone_slot_down_left_in_pixel * img_down_mm_per_pixel_ratio, 2)
                delta_distance_phone_slot_down_right_in_mm = np.round(delta_distance_phone_slot_down_right_in_pixel * img_down_mm_per_pixel_ratio, 2)
                cv2.putText(image_debug, str(delta_distance_phone_slot_down_left_in_mm) +"mm", (img_down_center_left[0] + 20, img_down_center_left[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)
                cv2.putText(image_debug, str(delta_distance_phone_slot_down_right_in_mm) +"mm", (img_down_center_right[0] - 60, img_down_center_right[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)
                if (delta_distance_phone_slot_down_left_in_mm > 5 and delta_distance_phone_slot_down_right_in_mm > 5):
                    result_dict["phone_3"] = "not_misaligned"
                else: result_dict["phone_3"] = "misaligned"
            else: result_dict["phone_3"] = "not_misaligned"
            
            path_image_debug = os.path.join(ml_debug_path, 'image_debug_3.jpg')
            cv2.imwrite(path_image_debug, image_debug)
            
            print("result_dict", result_dict)
            self.log.info('result_dict = {}'.format(result_dict))
            self.log.info('Draw debug image successful')
        
            return result_dict, image_debug
        
        except Exception as e:
            self.log.exception("Exception = {}".format(e))
            print(e)
            self.log.exception('result_dict = {}'.format(result_dict))
            return result_dict, image_org


if __name__ == '__main__':
    time_start = time.time()
    side_wall = 'right'
    number_slots_detector = 3
    image_path = "/home/greystone/StorageWall/debug/1726638461566_2639_detect_phone_slot_org.jpg"
    model_path = 'model_object_detect_magnetic_yolov8.onnx'
    labelmap_path = "/home/greystone/StorageWall/model_template/Model_Triton/model_object_detect_magnetic_yolov8/label.pbtxt"
    ml_debug_path = "/home/greystone/StorageWall/debug"
    position_have_phone = (None, None, (412, 554, 752, 664))
    object_detector = CalibMagnetArea_3slots(model_path, labelmap_path, 0.70)
    image = cv2.imread(image_path)
    image_debug = cv2.imread("/home/greystone/StorageWall/debug/1726638461566_2639_detect_phone_slot_org.jpg")
    result_dict, image_debug = object_detector.run(image, image_debug, side_wall, ml_debug_path, number_slots_detector, position_have_phone)
    time_elapsed = time.time() - time_start
    print("time_elapsed = ",time_elapsed)
