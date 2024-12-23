import cv2
import os
import time
import subprocess
import datetime

if os.getenv('GENICAM_GENTL64_PATH') is None:
    os.environ['GENICAM_GENTL64_PATH'] = '/usr/lib/ids/cti'

from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl

def get_index_cam_by_serial(serial, FILTER = "ID_SERIAL_SHORT="):
    """ Get index of camera tracking by serial

    Args:
        serial: serial of camera tracking  
        FILTER: Defaults to "ID_SERIAL_SHORT=".

    Returns:
        index of camera tracking
    """
    p = subprocess.run('v4l2-ctl --list-devices', stdout=subprocess.PIPE, shell=True)
    output = p.stdout.decode("utf-8")
    number_cam_id = []
    index = output.find("/dev/video")
    while index != -1:
        number_cam_id.append(int(output[index + 10]))
        output = output[index + 11:]
        index = output.find("/dev/video")
    
    for cam_id in number_cam_id:
        p = subprocess.run('udevadm info --query=all -n /dev/video{} | grep {}'.format(cam_id, FILTER), stdout=subprocess.PIPE, shell=True)
        output = p.stdout.decode("utf-8")
        output = output.split("=")[1].replace("\n", "")
        if output == serial:
            break
    return cam_id

def unique_str():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Extract individual components
    day = current_datetime.day
    hour = current_datetime.hour
    minute = current_datetime.minute
    second = current_datetime.second
    microsecond = current_datetime.microsecond // 1000  # Convert microseconds to milliseconds

    # Concatenate the components with no spaces or underscores and convert to lowercase
    formatted_string = f"{day}{hour}{minute}{second}{microsecond}".lower()
    return formatted_string

class BaseCamera:
    def __init__(self):
        self.is_opened = False
        self.camera = None

    def open(self):
        raise NotImplementedError("open method must be implemented in subclasses")
    
    def isOpened(self):
        return self.is_opened
    
    def get_width(self):
        raise NotImplementedError("get_width method must be implemented in subclasses")
    
    def get_height(self):
        raise NotImplementedError("get_height method must be implemented in subclasses")
    
    def get_fps(self):
        raise NotImplementedError("get_fps method must be implemented in subclasses")
    
    def set_exposure(self):
        raise NotImplementedError("set_exposure method must be implemented in subclasses")
    
    def set_gain(self):
        raise NotImplementedError("set_gain method must be implemented in subclasses")
    
    def set_fps(self):
        raise NotImplementedError("set_fps method must be implemented in subclasses")
    
    def read(self):
        raise NotImplementedError("read_frame method must be implemented in subclasses")

    def release(self):
        raise NotImplementedError("release method must be implemented in subclasses")


class See3CAM(BaseCamera):
    def __init__(self, serial_number):
        self.serial_number = serial_number
        self.id_camera = get_index_cam_by_serial(self.serial_number)
    
    def open(self):
        self.camera = cv2.VideoCapture()
        self.camera.open(self.id_camera, apiPreference=cv2.CAP_V4L2)
        if self.camera.isOpened():
            self.is_opened = True
            return True
    
    def get_width(self):
        return self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    def get_height(self):
        return self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def get_fps(self):
        return self.camera.get(cv2.CAP_PROP_FPS)
    
    def set_exposure(self, exposure):
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
        self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
    
    def set_gain(self, gain):
        self.camera.set(cv2.CAP_PROP_GAIN, gain)
        
    def set_fps(self, fps):
        self.camera.set(cv2.CAP_PROP_FPS, fps)
    
    def set_resolution(self, width, height):
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('U', 'Y', 'V', 'Y'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_focus(self, value):
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Set to 0 to disable autofocus
        os.system("v4l2-ctl -c focus_auto=0")
        time.sleep(1)
        if self.camera.get(cv2.CAP_PROP_AUTOFOCUS) == 0:
            # If autofocus is turned off (0), set the focus manually
            self.camera.set(cv2.CAP_PROP_FOCUS, value)
            return True
        else:
            return False
    
    def read(self):
        ret, frame = self.camera.read()
        return ret, frame

class IDSCamera(BaseCamera):
    def __init__(self):
        peak.Library.Initialize()
        self.device_manager = peak.DeviceManager.Instance()
        self.device_manager.Update()
        self.device_manager.Devices()
        self.flag_correct_color = False
        self.is_opened = False

    def open(self):
        self.camera = self.device_manager.Devices()[0]
        if self.camera is None:
            self.is_opened = False
            print("Not found any device IDS")
            return
        
        self.m_device = self.camera.OpenDevice(peak.DeviceAccessType_Control)
        
        if self.m_device is None:
            self.is_opened = False
            print("Cannot open camera IDS")
            return
            
        self.m_image_transformer_ipl = ids_peak_ipl.ImageTransformer()
        self.m_gamma_corrector = ids_peak_ipl.GammaCorrector()
        self.m_color_corrector_ipl = ids_peak_ipl.ColorCorrector()
        self.m_color_corrector_factor = ids_peak_ipl.ColorCorrectionFactors()
        self.m_color_corrector_factor.factorRR = 1.38
        self.m_color_corrector_factor.factorGR = -0.44
        self.m_color_corrector_factor.factorBR = 0.05
        
        self.m_color_corrector_factor.factorRG = -0.22
        self.m_color_corrector_factor.factorGG = 1.34
        self.m_color_corrector_factor.factorBG = -0.11
        
        self.m_color_corrector_factor.factorRB = 0.13
        self.m_color_corrector_factor.factorGB = -0.93
        self.m_color_corrector_factor.factorBB = 1.80
        self.m_color_corrector_ipl.SetColorCorrectionFactors(self.m_color_corrector_factor)
        
        self.m_node_map_remote_device = self.m_device.RemoteDevice().NodeMaps()[0]
      
        self.is_opened = True
        
    def get_width(self):
        return self.m_node_map_remote_device.FindNode("Width").Value()
    
    def get_height(self):
        return self.m_node_map_remote_device.FindNode("Height").Value()
    
    def get_fps(self):
        return self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").Value()
    
    def set_exposure(self, value):
        self.m_node_map_remote_device.FindNode("ExposureTime").SetValue(value)
    
    def set_gain(self, gain):
        self.m_node_map_remote_device.FindNode("Gain").SetValue(gain)
    
    def set_fps(self, value):
        min_frame_rate = 0
        max_frame_rate = 0
        inc_frame_rate = 0
        
        # Get frame rate range. All values in fps.
        min_frame_rate = self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").Minimum()
        max_frame_rate = self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").Maximum()
        
        if self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").HasConstantIncrement():
            inc_frame_rate = self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").Increment()
        else:
            # If there is no increment, it might be useful to choose a suitable increment for a GUI control element (e.g. a slider)
            inc_frame_rate = 0.1
        
        # Set frame rate to maximum
        self.m_node_map_remote_device.FindNode("AcquisitionFrameRate").SetValue(max_frame_rate)
        
    def set_decimation(self, vec, hoz):
        self.m_node_map_remote_device.FindNode("DecimationSelector").SetCurrentEntry("Sensor")
        self.m_node_map_remote_device.FindNode("DecimationHorizontal").SetValue(hoz)
        self.m_node_map_remote_device.FindNode("DecimationVertical").SetValue(vec)

    def set_gamma(self, gamma):
        self.m_gamma_corrector.SetGammaCorrectionValue(gammaValue=gamma)
                
    def set_roi(self, x, y, width, height):   
        # Get the minimum ROI and set it. After that there are no size restrictions anymore
        x_min = self.m_node_map_remote_device.FindNode("OffsetX").Minimum()
        y_min = self.m_node_map_remote_device.FindNode("OffsetY").Minimum()
        w_min = self.m_node_map_remote_device.FindNode("Width").Minimum()
        h_min = self.m_node_map_remote_device.FindNode("Height").Minimum()

        self.m_node_map_remote_device.FindNode("OffsetX").SetValue(x_min)
        self.m_node_map_remote_device.FindNode("OffsetY").SetValue(y_min)
        self.m_node_map_remote_device.FindNode("Width").SetValue(w_min)
        self.m_node_map_remote_device.FindNode("Height").SetValue(h_min)
        
        # Get the maximum ROI values
        x_max = self.m_node_map_remote_device.FindNode("OffsetX").Maximum()
        y_max = self.m_node_map_remote_device.FindNode("OffsetY").Maximum()
        w_max = self.m_node_map_remote_device.FindNode("Width").Maximum()
        h_max = self.m_node_map_remote_device.FindNode("Height").Maximum()

        
        if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
            return False
        elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
            return False
        else:
            # Now, set final AOI
            self.m_node_map_remote_device.FindNode("OffsetX").SetValue(x)
            self.m_node_map_remote_device.FindNode("OffsetY").SetValue(y)
            self.m_node_map_remote_device.FindNode("Width").SetValue(width)
            self.m_node_map_remote_device.FindNode("Height").SetValue(height)

    def start_stream(self):
        data_streams = self.m_device.DataStreams()
        if data_streams.empty():
            # no data streams available
            return False
        
        self.m_dataStream = self.m_device.DataStreams()[0].OpenDataStream()
        
        if self.m_dataStream:
            # Flush queue and prepare all buffers for revoking
            self.m_dataStream.Flush(peak.DataStreamFlushMode_DiscardAll)

            # Clear all old buffers
            for buffer in self.m_dataStream.AnnouncedBuffers():
                self.m_dataStream.RevokeBuffer(buffer)

            payload_size = self.m_node_map_remote_device.FindNode("PayloadSize").Value()

            # Get number of minimum required buffers
            num_buffers_min_required = self.m_dataStream.NumBuffersAnnouncedMinRequired()

            # Alloc buffers
            for count in range(num_buffers_min_required):
                buffer = self.m_dataStream.AllocAndAnnounceBuffer(payload_size)
                self.m_dataStream.QueueBuffer(buffer)
        
        self.m_dataStream.StartAcquisition(peak.AcquisitionStartMode_Default, peak.DataStream.INFINITE_NUMBER)
        self.m_node_map_remote_device.FindNode("TLParamsLocked").SetValue(1)
        self.m_node_map_remote_device.FindNode("AcquisitionStart").Execute()

    def read(self):
        buffer = self.m_dataStream.WaitForFinishedBuffer(5000)
        
        image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
            buffer.PixelFormat(),
            buffer.BasePtr(),
            buffer.Size(),
            buffer.Width(),
            buffer.Height()
        )
        
        # Create IDS peak IPL image for debayering and convert it to RGBa8 format
        image_processed = image.ConvertTo(ids_peak_ipl.PixelFormatName_BGR8, ids_peak_ipl.ConversionMode_Fast)
        self.m_gamma_corrector.ProcessInPlace(image_processed)
        # if self.flag_correct_color == True:
        # self.m_color_corrector_ipl.ProcessInPlace(image_processed)
        image_np = image_processed.get_numpy_3D()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

        ret = True
        if image_np is None:
            ret = False
        
        # Queue buffer again
        self.m_dataStream.QueueBuffer(buffer)

        # return ret, image_np, image_processed
        return ret, image_np

class CameraFactory:
    @staticmethod
    def create_camera(camera_type, serial_number=None):
        if camera_type == "See3CAM":
            return See3CAM(serial_number)
        elif camera_type == "IDSCamera":
            return IDSCamera()
        else:
            raise ValueError("Unsupported camera type")
        
# Example usage:
if __name__ == "__main__":
   
    get_view = False
    get_frame = True
    test_FPS = False
    
    path_frame_folder = "/home/greystone/LongVu/video"
    os.makedirs(path_frame_folder, exist_ok=True)

    # Create camera factory
    factory = CameraFactory()
    
    # # Create See3CAM
    # see3cam = factory.create_camera("See3CAM")
    # see3cam.initialize()
    # see3cam.open()
    # frame_see3cam = see3cam.read_frame()
    # # Process the frame or display it as needed
    # see3cam.release()

    # Create IDS camera
    ids_camera = factory.create_camera("IDSCamera")
    ids_camera.open()
    ids_camera.set_roi(136, 0, 1200, 600)
    ids_camera.set_exposure(30000)
    ids_camera.set_gain(2.86)
    ids_camera.set_gamma(1.0)
    ids_camera.set_fps(30)

    ids_camera.start_stream()
        
    start_time = time.time()
    count_avg = 0
    total_time_avg = 0
    camera_fps = 0
    frame_count = 0
    while True:
        # time.sleep(0.005)
        # start_time = time.time()

        ret, frame_ids = ids_camera.read()
        
        if ret or frame_ids is not None:
            frame_count += 1
            FPS = frame_count / (time.time() - start_time)
            
            if get_view:
                frame_ids_resize = cv2.resize(frame_ids, (640,480))
                cv2.imwrite(os.path.join(path_frame_folder, 'frame_org.jpg'), frame_ids)
                cv2.imwrite(os.path.join(path_frame_folder, 'frame_resize.jpg'), frame_ids_resize)
                break
            elif get_frame:
                cv2.imshow('frame', frame_ids)
                end_time = time.time()
                delta_time = end_time - start_time
                key = cv2.waitKey(1) & 0xFF
                # break on pressing q
                if key == ord('q'):
                    break
                elif key == 32 or (True and delta_time >= 0.2):
                    start_time = time.time()
                    image_file_name = "frame_{}.jpg".format(unique_str())
                    path_to_img = os.path.join(path_frame_folder, image_file_name)
                    print("path_to_img = {}".format(path_to_img))
                    cv2.imwrite(path_to_img, frame_ids)         
            
            else: 
                cv2.imshow('frame', frame_ids)
                key = cv2.waitKey(1) & 0xFF
                # break on pressing q
                if key == ord('q'):
                    break
                
                count_avg += 1
                total_time_avg += (time.time() - start_time)
                if count_avg == 10:
                    camera_fps = 1 / (total_time_avg/10)
                    count_avg = 0
                    total_time_avg = 0
                start_time = time.time()
                
                print("FPS = ", camera_fps)
            
    # Process the frame or display it as needed
    # ids_camera.release()
    print("CLOSE CAMERA IDS")