### Main Feature
ML-Sub-Storage-Wall is a python subapp whose main function is communicate with Grey software and display back on Storage Wall software.
  - Detail feature of subapp:
    - Barcode Scanner
    - Detect phone
    - Calibrate phone slot
### Install
- Set up and activate the environment for the new device
- cd to respository_folder
```bash
cd <respository_folder>
conda env create -f ml_subapp_environment.yml
```
```bash
conda activate ml_subapp
```
  - Load docker from .tar image for the new device
cd to respository_folder
```bash
cd /home/greystone/StorageWall/model_template/ScanBarcode
```
```bash
sudo docker load -i docker_barcode.tar.gz
```
```bash
sudo docker run -e LD_PRELOAD=/usr/local/lib/faketime/libfaketime.so.1 \
-e FAKETIME="2021-05-12 10:30:00" \
-it -d --network none --name docker_barcode -v/home/greystone:/home/greystone docker_barcode:test
```

### Setup directory
1. ONNX Models
A minimal ONNX model definition looks like this:
- |folder-path # need to create manual
  - |model-name # folder is created automatically when run subapp
    - |model-name.onnx   # use for ONNX model 
    - |model-name.pt     # used for the original Yolo model


### Run subapp python
    $ cd <respository_folder>
    $ conda activate manage_triton_server
    $ python main_app.py -m 1

### All model platform support in subApp
- YOLOv8:
   - yolov8_detection
   - yolov8_segmentation
