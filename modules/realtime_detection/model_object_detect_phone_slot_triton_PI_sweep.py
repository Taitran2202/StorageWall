import datetime
import json
import logging
import os
import time
from typing import Tuple

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from globals import GV

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
        self.result_have_phone = 'have_phone'
        self.result_none_phone = 'none_phone'
        self.tag_log = '[PhoneSlotObjectDetector]'
        self.quantity_row = 54
        self.quantity_column = 24
        self.offset_x_min = 0
        self.offset_y_min = 0
        self.offset_x_max = 3280
        self.offset_y_max = 2455
    
    def log_msg(self, msg, text_log_file):
        # Get current timeestamp
        current_timestamp = datetime.datetime.now()
        formatted_string = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = self.tag_log + '-' + formatted_string + ':' + msg
        # Log message
        text_log_file.write(msg)
    
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
    
    def draw_area_box(self, image, x_from, y_from, x_to, y_to, color_bgr, size=0.005, line_type=cv2.LINE_AA, is_copy=True):
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

class Object_Detection_YOLOv8_PI_SWEEP(YOLOv8_Triton):

    def __init__(self, model_path, labelmap_path, threshold) -> None:
        YOLOv8_Triton.__init__(self, model_path, labelmap_path, threshold)

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
            print("gain: ",gain)
            pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
                (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
            print("pad: ", pad)
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

        t = time.time()

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
     
    def find_position_phone_slot(self, offset_x_cur, offset_y_cur):
        step_x = (self.offset_x_max - self.offset_x_min)/(self.quantity_column - 1)
        step_y = (self.offset_y_max - self.offset_y_min)/(self.quantity_row - 1)

        slot_x_cur = round((offset_x_cur - self.offset_x_min + step_x)/step_x)
        slot_y_cur = round((offset_y_cur - self.offset_y_min + step_y)/step_y)
        
        delta_slot_x_cur_in_mm = offset_x_cur - self.offset_x_min - ((slot_x_cur-1)*step_x)
        delta_slot_y_cur_in_mm = offset_y_cur - self.offset_y_min - ((slot_y_cur-1)*step_y)
        
        delta_slot_x_cur_in_pixel = round((delta_slot_x_cur_in_mm / step_x) * 925)
        delta_slot_y_cur_in_pixel = round((delta_slot_y_cur_in_mm / step_y) * 235)
        
        return slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel
    
    def check_area_detect(self, x_from, x_to, y_from, y_to, delta_slot_x_cur_in_pixel,delta_slot_y_cur_in_pixel):
        x_from = x_from + delta_slot_x_cur_in_pixel
        x_to = x_to + delta_slot_x_cur_in_pixel
        y_from = y_from + delta_slot_y_cur_in_pixel
        y_to = y_to + delta_slot_y_cur_in_pixel
        
        if x_from < 0:
            x_from = 0
        if x_to > 1200:
            x_to = 1200
        if y_from < 0:
            y_from = 0
        if y_to > 600:
            y_to = 600
        return x_from, x_to, y_from, y_to
    
    def detect(self, image):
        self.log.info('In function detect!')
        # Ping to model on triton
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
        # print("result flag:", boxes.flags)
        masks = None

        #print("boxes flag:", boxes.flags)
        print("boxes shape: ", boxes.shape)
        
        # Postprocess for bbox
        detections = self.non_max_suppression(boxes, self.conf_thres)
        detections = np.array(detections)
        print("detection shape: ", len(detections))
        
        if len(detections) == 0:
            self.log.info("Object empty!")
            return None
        
        detections[:, :4] = self.scale_boxes((640, 640), detections[:, :4], (h, w)).round()
        
        # Save result bbox
        bboxes = []
        for bbox in detections:
            class_name = self.cls_names.get(int(bbox[5]) + 1)
            score = round(float(bbox[4]), 2)
            bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), class_name, score])
        self.log.info('End function detect!')
        return bboxes
    
    def post_process(self, image, bboxes, positionToDetect, position_of_phoneslot_cur, min_max_wall):
            # Write the text to the file
        result = self.result_none_phone
        self.log.info('In function post-proceess!')
        phone_list = []
        debug_img = image
        slot_id = None
        self.offset_x_min, self.offset_x_max, self.offset_y_min, self.offset_y_max = min_max_wall
        
        if positionToDetect == 'left':
            self.log.info('Open file config for detect the phone in the left wall!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_left_wall_from', 350)
            y_from = data.get('y_detect_phone_left_wall_from', 170)
            x_to = data.get('x_detect_phone_left_wall_to', 850)
            y_to = data.get('y_detect_phone_left_wall_to', 280)
            self.quantity_row = data.get('quantity_row')
            self.quantity_column = data.get('quantity_column')
        elif positionToDetect == 'right':
            self.log.info('Open file config for detect the phone in the right wall!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_right_wall_from', 315)
            y_from = data.get('y_detect_phone_right_wall_from', 190)
            x_to = data.get('x_detect_phone_right_wall_to', 775)
            y_to = data.get('y_detect_phone_right_wall_to', 315)
            self.quantity_row = data.get('quantity_row')
            self.quantity_column = data.get('quantity_column')
        elif positionToDetect == 'buffer':
            self.log.info('Open file config for detect the phone in the buffer!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_kangaroo_from', 300)
            y_from = data.get('y_detect_phone_kangaroo_from', 30)
            x_to = data.get('x_detect_phone_kangaroo_to', 900)
            y_to = data.get('y_detect_phone_kangraroo_to', 455)
        elif positionToDetect == 'input_dock':
            self.log.info('Open file config for detect the phone in the Input dock!')
            with open('/home/greystone/StorageWall/apps/ml-subapp/appPython/config.json', 'r') as json_file:
                data = json.load(json_file)
            x_from = data.get('x_detect_phone_input_dock_from', 1450)
            y_from = data.get('y_detect_phone_input_dock_from', 515)
            x_to = data.get('x_detect_phone_input_dock_to', 2750)
            y_to = data.get('y_detect_phone_input_dock_to', 2815)
        else:
            self.log.info('ERROR: Require key positionToDetect from SW!')
        
        position_X_cur, position_Z_cur = position_of_phoneslot_cur
        
        slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel = self.find_position_phone_slot(position_X_cur, position_Z_cur)
        self.log.info('slot_x_cur = {}, slot_y_cur = {}, delta_slot_x_cur_in_pixel = {}, delta_slot_y_cur_in_pixel = {}'.format(slot_x_cur, slot_y_cur, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel))
        
        x_from, x_to, y_from, y_to = self.check_area_detect(x_from, x_to, y_from, y_to, delta_slot_x_cur_in_pixel, delta_slot_y_cur_in_pixel)
        
        if positionToDetect == 'left':
            side_cur = 'LS'
        elif positionToDetect == 'right':
            side_cur = 'RS'
            
        formatted_slot_x_cur = str(slot_x_cur).zfill(2)
        formatted_slot_y_cur = str(slot_y_cur).zfill(2)
        slot_id = side_cur + 'R' + formatted_slot_y_cur + 'C' + formatted_slot_x_cur
        
        cv2.putText(debug_img, f"X: {position_X_cur}, Z: {position_Z_cur}", (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
        cv2.putText(debug_img, slot_id, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
        
        debug_img, location_area_detect = self.draw_area_box(debug_img, x_from, y_from, x_to, y_to, color_bgr = [0, 0, 255], is_copy = True)
        
        is_has_phone = False
        overlap_percentage_target = 0.00
        img_check_area = np.zeros((debug_img.shape[0], debug_img.shape[1], 1), dtype=np.uint8)
        cv2.rectangle(img_check_area, (location_area_detect[0], location_area_detect[1]), (location_area_detect[0] + location_area_detect[2], location_area_detect[1] + location_area_detect[3]), [255], -1)

        if bboxes is None:
            result = self.result_none_phone
            return result, slot_id, debug_img

        mask = np.zeros(image.shape, dtype=np.uint8)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, cls_name, score = bbox
            if cls_name == 'phone':
                phone_list.append(bbox)
                cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 225, 225), -1)
                cv2.putText(debug_img, str(cls_name), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(debug_img, '{score}%'.format(score=int(score*100)), (xmin, ymax-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (55, 255, 100), 1, cv2.LINE_AA)
        
        for phone in phone_list:
            # check position of phone cross the line
            xmin, ymin, xmax, ymax, cls_name, score = phone
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            cur_img = np.zeros((debug_img.shape[0], debug_img.shape[1], 1), dtype=np.uint8)
            cv2.rectangle(cur_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), [255], -1)
            img_dst = cv2.bitwise_and(img_check_area, cur_img)
            nonZeroVal = cv2.countNonZero(img_dst)
            overlap_percentage = (nonZeroVal * 100) / (location_area_detect[2]*location_area_detect[3])
            self.log.info("overlap_percentage = {}".format(overlap_percentage))
            if overlap_percentage > 25:
                overlap_percentage = np.round(overlap_percentage,2)
                overlap_percentage_target = overlap_percentage
                is_has_phone = True
                cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), [255, 0, 0], 5)
        msg = "overlap_percentage = " + str(overlap_percentage_target) + "%"
        cv2.putText(debug_img, msg, (0,200), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)

        if is_has_phone:
            result = self.result_have_phone
            self.log.info('Found overlapped phone. Return have_phone')
        else:
            result = self.result_none_phone
            self.log.info('Not found overlapped phone. Return no_phone')

        
        return result, slot_id, debug_img

    
    def run(self, original_image, positionToDetect, position_of_phoneslot_cur, min_max_wall):
        # init logger
        time_start = time.time()
        try:
            result = self.result_none_phone
            
            if original_image is None:
                self.log.info('Error: Image not found!')
                return result, None, None
            else:
                self.log.info('Image opened!')
            
            # Detect
            self.log.info('Start run function detect!')
            object_list = self.detect(original_image)             
            self.log.info('End run function detect!')
            
            # Print out current object
            msg  = 'Current object list: ' + str(object_list) + '\n'
            self.log.info(f'Current object list: {object_list}')
            # if object_list is None:
            #     result = self.result_none_phone
            #     return result, None, original_image
            
            # post process
            self.log.info('Start run function post-process!')
            result, slot_id, debug_img = self.post_process(original_image, object_list, positionToDetect, position_of_phoneslot_cur, min_max_wall)
            self.log.info('End run function post-process!')
            
            self.log.info(f"Result = {result}")

            # draw debug
            cv2.putText(debug_img, result, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
            
            # Close file
            self.log.info('Draw debug image successful')
            
            return result, slot_id, debug_img

        except Exception as e:
            self.log.info(f'ERROR: {e}')
            result = self.result_none_phone
            self.log.info(f'Result = {result}')
            return result, None, None

if __name__ == "__main__":
    time_start = time.time()
    positionToDetect = 'left'
    model_path = "./model_object_detect_phone_slot.onnx"
    labelmap_path = "/home/greystone/StorageWall/model_template/Model_Triton/model_object_detect_phone_slot/label.pbtxt"
    threshold = 0.7
    model = Object_Detection_YOLOv8_PI_SWEEP(model_path, labelmap_path, threshold)

    #image_path = "/home/greystone/StorageWall/image_debug/W01LSR01C01_detect_phone_slot_org.jpg"
    image_path = "/home/greystone/StorageWall/image_debug/W01W01LSR07C04_20240507141854765.png"
    result, image_debug = model.run(image_path, positionToDetect)
    print("image shape = ", image_debug.shape)
    cv2.imwrite("/home/greystone/StorageWall/image_debug/pi_debug.png", image_debug)
    time_stop = time.time()
    print("time elapsed :", time_stop - time_start)

