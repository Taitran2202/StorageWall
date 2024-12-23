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
        self.log = logging.getLogger(__name__)
        self.model_path = model_path
        self.cls_names = self.load_labelmap(labelmap_path)
        self.conf_thres = threshold
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
        image = image.copy() if is_copy else image
        h, w = image.shape[:2]
        if size > 0:        
            short_edge = min(h, w)
            thickness = int(short_edge * size)
            thickness = 1 if thickness <= 0 else thickness
        else:
            thickness = -1
        cv2.rectangle(img=image, pt1=(x_from, y_from), pt2=(x_to, y_to), color=color_bgr, thickness=thickness, lineType=line_type, shift=0)
        location = (x_from, y_from, x_to - x_from, y_to - y_from)
        return image, location

class Detect_Middle_Magnet(YOLOv8_Triton):
    def __init__(self, model_path, labelmap_path, threshold) -> None:
        YOLOv8_Triton.__init__(self, model_path, labelmap_path, threshold)

    def log_msg(self, msg, text_log_file):
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ':' + msg
        text_log_file.write(msg)
        
    def load_coordinates_to_crop(Self, side):
        with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
            data = json.load(json_file)
        if side == 'left':
            position_crop_magnet_right =    [   data.get("y_from_wall_left_2"),
                                                data.get("y_to_wall_left_2"),
                                                data.get("x_from_wall_left_2"),                                               
                                                data.get("x_to_wall_left_2")]    
            position_crop_magnet_left =     [   data.get("y_from_wall_left_1"),
                                                data.get("y_to_wall_left_1"),
                                                data.get("x_from_wall_left_1"),                                               
                                                data.get("x_to_wall_left_1")]
            org_position = (data.get("x_center_point_left"), data.get("y_center_point_left"), data.get("rotation_org_left"))
        elif side == 'right':  
            position_crop_magnet_right =    [   data.get("y_from_wall_right_2"),
                                                data.get("y_to_wall_right_2"),
                                                data.get("x_from_wall_right_2"),                                               
                                                data.get("x_to_wall_right_2")]    
            position_crop_magnet_left =     [   data.get("y_from_wall_right_1"),
                                                data.get("y_to_wall_right_1"),
                                                data.get("x_from_wall_right_1"),                                               
                                                data.get("x_to_wall_right_1")]
            org_position = (data.get("x_center_point_right"), data.get("y_center_point_right"), data.get("rotation_org_right"))
        elif side == 'left_exception':  
            position_crop_magnet_right =    [   data.get("y_from_wall_right_2"),
                                                data.get("y_to_wall_right_2"),
                                                data.get("x_from_wall_right_2"),                                               
                                                data.get("x_to_wall_right_2")]    
            position_crop_magnet_left =     [   data.get("y_from_wall_right_1"),
                                                data.get("y_to_wall_right_1"),
                                                data.get("x_from_wall_right_1"),                                               
                                                data.get("x_to_wall_right_1")]
            org_position = (data.get("x_center_point_right"), data.get("y_center_point_right"), data.get("rotation_org_right"))
        elif side == 'right_exception':  
            position_crop_magnet_right =    [   data.get("y_from_wall_right_2"),
                                                data.get("y_to_wall_right_2"),
                                                data.get("x_from_wall_right_2"),                                               
                                                data.get("x_to_wall_right_2")]    
            position_crop_magnet_left =     [   data.get("y_from_wall_right_1"),
                                                data.get("y_to_wall_right_1"),
                                                data.get("x_from_wall_right_1"),                                               
                                                data.get("x_to_wall_right_1")]
            org_position = (data.get("x_center_point_right"), data.get("y_center_point_right"), data.get("rotation_org_right"))
        elif side == 'center_exception':  
            position_crop_magnet_right =    [   data.get("y_from_wall_right_2"),
                                                data.get("y_to_wall_right_2"),
                                                data.get("x_from_wall_right_2"),                                               
                                                data.get("x_to_wall_right_2")]    
            position_crop_magnet_left =     [   data.get("y_from_wall_right_1"),
                                                data.get("y_to_wall_right_1"),
                                                data.get("x_from_wall_right_1"),                                               
                                                data.get("x_to_wall_right_1")]
            org_position = (data.get("x_center_point_right"), data.get("y_center_point_right"), data.get("rotation_org_right")) 
        offset_position = (data.get("offset_left"), data.get("offset_right"), data.get("offset_top"), data.get("offset_bot"), data.get("offset_rotation"))
        return position_crop_magnet_left, position_crop_magnet_right, org_position, offset_position
    
    def crop_img(self, image_org, position_crop_magnet_left, position_crop_magnet_right):
        image_crop_right = image_org[position_crop_magnet_right[0]:position_crop_magnet_right[1], position_crop_magnet_right[2]:position_crop_magnet_right[3], :]
        image_crop_left = image_org[position_crop_magnet_left[0]:position_crop_magnet_left[1], position_crop_magnet_left[2]:position_crop_magnet_left[3], :]
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
            x = x[xc[xi]]  # confidence
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
            if merge and (1 < n < 3E3): 
                iou = self.box_iou(boxes[keep_boxes], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    keep_boxes = keep_boxes[iou.sum(1) > 1]  # require redundancy
            for k in keep_boxes:
                output.append(x[k])
        return output
    def detect(self, image):
        self.log.info('In function detect!')
        self.log.info('model_name_on_triton: {}'.format(self.model_name_on_triton))
        is_model_ready = self._load_model_ready()
        if is_model_ready is False:
            self.log.warning("triton model {} is not ready; exit".format(self.model_name_on_triton))
            return []
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_image = self.preprocess_image(image)
        input_tensor = self.image_to_tensor(preprocessed_image)
        inputs = [grpcclient.InferInput('images', input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)
        outputs = [grpcclient.InferRequestedOutput('output0')]
        result = self.triton_client.infer(model_name=self.model_name_on_triton, inputs=inputs, outputs=outputs)
        boxes = []
        boxes = np.array(result.as_numpy('output0'))
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
            bboxes.append([class_name, score, int(bbox[0]) , int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])])
            
            current_object['label'] = self.cls_names.get(int(bbox[5])+1)
            current_object['conf'] = round(float(bbox[4]), 2)
            current_object['x'] = int((int(bbox[0]) + int(bbox[2]))/2)
            current_object['y'] = int((int(bbox[1]) + int(bbox[3]))/2)
            current_object['w'] = int(bbox[2]) - int(bbox[0])
            current_object['h'] = int(bbox[3]) - int(bbox[1])
            if (current_object['conf'] > self.conf_thres) and (current_object['label'] == 'magnetic_dark'):
                object_list.append(current_object)
        return object_list
    
    def run(self, path_image, image_debug, side, ml_debug_path):
        result_dict, position_magnetic_image_org = {}, [(0, 0), (0, 0)]
        image_org = cv2.imread(path_image)
        start_crop_magnetic_image = time.time()
        try:
            position_crop_left, position_crop_right, org_position, offset_position = self.load_coordinates_to_crop(side)
            image_crop_left, image_crop_right = self.crop_img(image_org, position_crop_left, position_crop_right)
            cv2.imwrite(os.path.join(ml_debug_path, "image_crop_left.png"), image_crop_left)
            self.log.info(ml_debug_path)
            cv2.imwrite(os.path.join(ml_debug_path, "image_crop_right.png"), image_crop_right)
            self.log.info('End run function crop!')
            
            cv2.rectangle(image_debug, (position_crop_left[2], position_crop_left[0]), (position_crop_left[3],position_crop_left[1]) , (0, 255, 255), 2)
            cv2.rectangle(image_debug, (position_crop_right[2], position_crop_right[0]), (position_crop_right[3], position_crop_right[1]), (0, 255, 255), 2)
            self.log.info("Time processing crop magnetic image:{} ".format(time.time() - start_crop_magnetic_image))
            self.log.info('Start run function detect!')
            start_time_detect_magnet_left = time.time()
            object_list_middle_left = self.detect(image_crop_left)
            self.log.info("Time to detect magnetic left:{} ".format(time.time()- start_time_detect_magnet_left))
            try:
                left_x_list = [item['x'] for item in object_list_middle_left]
                left_y_list = [item['y'] for item in object_list_middle_left]
                x_left = max(left_x_list)
                y_left = left_y_list[left_x_list.index(x_left)]
            except Exception as e:
                x_left = 160
                y_left = 80
            start_time_detect_magnet_right = time.time()
            object_list_middle_right = self.detect(image_crop_right)
            self.log.info("Time to detect magnetic right:{} ".format(time.time()- start_time_detect_magnet_right))
            try:
                right_x_list = [item['x'] for item in object_list_middle_right]
                right_y_list = [item['y'] for item in object_list_middle_right]
                x_right = min(right_x_list)
                y_right = right_y_list[right_x_list.index(x_right)]
            except Exception as e:
                x_right = 160
                y_right = 80
            start_time_calculate_deviation = time.time()
            position_magnetic_image_org = ((x_left + position_crop_left[2], y_left + position_crop_left[0]), (x_right + position_crop_right[2], y_right + position_crop_right[0]))
            cv2.circle(image_debug, (x_left + position_crop_left[2], y_left + position_crop_left[0]), 5, (0, 0, 255), -1)
            cv2.circle(image_debug, (x_right + position_crop_right[2], y_right + position_crop_right[0]), 5, (0, 0, 255), -1)
            
            rotation_gripper = math.degrees(math.atan2((y_right + position_crop_right[0]) - (y_left + position_crop_left[0]), (x_right + position_crop_right[2]) - (x_left + position_crop_left[2])))
            center_point =  [(position_magnetic_image_org[0][0]+ position_magnetic_image_org[1][0])/2, (position_magnetic_image_org[0][1]+ position_magnetic_image_org[1][1])/2]
            
            result_dict["left"]     = round((org_position[0]    - center_point[0])*0.16178, 2)      if center_point[0]  < org_position[0] - offset_position[0]      else 0
            result_dict["right"]    = round((center_point[0]    - org_position[0])*0.16178, 2)      if center_point[0]  > org_position[0] + offset_position[1]      else 0
            result_dict["top"]      = round((org_position[1]    - center_point[1])*0.16178, 2)      if center_point[1]  < org_position[1] - offset_position[2]      else 0
            result_dict["bot"]      = round((center_point[1]    - org_position[1])*0.16178, 2)      if center_point[1]  > org_position[1] + offset_position[3]      else 0
            result_dict["cw"]       = round(rotation_gripper    - org_position[2],          2)      if rotation_gripper > org_position[2] + offset_position[4]      else 0
            result_dict["ccw"]      = round(org_position[2]     - rotation_gripper,         2)      if rotation_gripper < org_position[2] - offset_position[4]      else 0
            cv2.putText(image_debug, ("left deviated: "     + str(result_dict["left"])  + " mm"),       (10,800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_debug, ("right deviated: "    + str(result_dict["right"]) + " mm"),       (10,830), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_debug, ("top deviated: "      + str(result_dict["top"])   + " mm"),       (10,860), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_debug, ("bot deviated: "      + str(result_dict["bot"])   + " mm"),       (10,890), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_debug, ("cw deviated: "       + str(result_dict["cw"])    + " degrees"),  (10,920), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_debug, ("ccw deviated: "      + str(result_dict["ccw"])   + " degrees"),  (10,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.log.info('result_dict = {}'.format(result_dict))
            cv2.imwrite(os.path.join(ml_debug_path, path_image.split('/')[-1].split('.')[0] + "_image_position_of_gripper.png"), image_debug)
            self.log.info('Draw debug magnetic image successful')
            self.log.info("Time processing calculate deviation:{}".format(time.time() - start_time_calculate_deviation))
            
            return result_dict, image_debug, position_magnetic_image_org
        except Exception as e:
            self.log.exception("Exception = {}".format(e))
            print(e)
            self.log.exception('result_dict = {}'.format(result_dict))
            return result_dict, image_debug, position_magnetic_image_org
        