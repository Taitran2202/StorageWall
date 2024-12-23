'''
python3 modules/detect_object_phone_slot/script_convert_voc_2_yolo.py \
    dataset/model_object_detect_phone_slot/dataset/train/xmls \
        dataset/model_object_detect_phone_slot/dataset/train/images
'''
import sys
from pylabel import importer

print('import successfuly!')

# init
path_to_annotations = sys.argv[1] # "dataset/model_object_detect_phone_slot/data/test/xmls"

#Identify the path to get from the annotations to the images 
path_to_images = sys.argv[2] # "dataset/model_object_detect_phone_slot/data/test/images"

dataset = importer.ImportVOC(path=path_to_annotations, path_to_images=path_to_images, name="Dataset")

print(dataset.df.head())

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")

# export to yolov5 format
dataset.export.ExportToYoloV5()

print('Export successfuly!')