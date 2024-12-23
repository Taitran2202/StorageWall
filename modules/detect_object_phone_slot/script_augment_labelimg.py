'''
python3 modules/detect_phone_slot/script_augment_labelme.py \
     dataset/model_detect_phone_slot/images/data2/val 100
'''
import sys
import os
import cv2
import imgaug.augmenters as iaa
from xml.dom import minidom
import xml.etree.ElementTree as ET
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Init
M = int(sys.argv[2])
src_path = sys.argv[1]
dst_path = 'dst'
image_files  = []
xml_files = []
image_tails = ["jpg","png","jpeg"]

def extract_bounding_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    bounding_boxes = []
    
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        label = obj.find('name').text
        
        bounding_boxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=label))
    
    return bounding_boxes

def export_to_xml(image_name, W, H, bounding_boxes, xml_file):
    root = ET.Element("annotations")

    # <folder>Downloads</folder>
    obj = ET.SubElement(root, "folder")
    obj.text = "Augmented"
    
    # <filename>augmented_0.jpg</filename>
    obj = ET.SubElement(root, "filename")
    obj.text = image_name
    
    # <path>/home/greystone/Downloads/augmented_0.jpg</path>
    image_path = '/path/to/'+image_name
    obj = ET.SubElement(root, "path")
    obj.text = image_path
    
    # <source>
	# 	<database>Unknown</database>
	# </source>
    obj = ET.SubElement(root, "source")
    sub_obj = ET.SubElement(obj, "database")
    sub_obj.text = "Unknown"
    
    # <size>
	# 	<width>1200</width>
	# 	<height>675</height>
	# 	<depth>3</depth>
	# </size>
    obj = ET.SubElement(root, "size")
    sub_obj = ET.SubElement(obj, "width")
    sub_obj.text = str(W)
    sub_obj = ET.SubElement(obj, "height")
    sub_obj.text = str(H)
    sub_obj = ET.SubElement(obj, "depth")
    sub_obj.text = str(3)
 
    # Add objects
    for bb in bounding_boxes:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = bb.label
        bbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bbox, "xmin")
        xmin.text = str(bb.x1)
        ymin = ET.SubElement(bbox, "ymin")
        ymin.text = str(bb.y1)
        xmax = ET.SubElement(bbox, "xmax")
        xmax.text = str(bb.x2)
        ymax = ET.SubElement(bbox, "ymax")
        ymax.text = str(bb.y2)

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(xml_file, "w") as f:
        f.write(xmlstr)

# instance of augmenter
my_augmenter = iaa.Sequential([
    iaa.Fliplr(0.25), # horizontal flips
    iaa.Flipud(0.25), # vertical flips
    iaa.Crop(percent=(0, 0.05)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.25,
        iaa.GaussianBlur(sigma=(0, 0.1))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.85, 1.15)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.9, 1.1), per_channel=0.05),
    # Apply rotate
    iaa.Rotate((-4,4))
    ], random_order=True) # apply augmenters in random order

if __name__ == "__main__":
    
    print('Import successful!')
    
    # Create debug path
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        
    # Check files
    files = os.listdir(src_path)
    
    # check image file
    for file in files:
        file_name = file.split(".")[0]
        file_tail = file.split(".")[-1]
        if file_tail in image_tails:
            image_files.append(file)
            xml_files.append(file_name + '.xml')
            
    # Calculate image_augmented per image
    N = len(image_files)
    k = int(M/N)
    
    for i in range(N):
        print(f'({i+1}/{N}) augment image={image_files[i]} xml_file={xml_files[i]}')
        
        file_name = image_files[i].split(".")[0] 
        image_path = os.path.join(src_path, image_files[i])
        xml_path = os.path.join(src_path, xml_files[i]) 
        
        # Read image
        img = cv2.imread(image_path)
        H, W, _ = img.shape
        
        # Read annotation
        bounding_boxes = extract_bounding_boxes(xml_path)
        
        # create list of augmented
        augmented_list = [my_augmenter(image = img, bounding_boxes = bounding_boxes) for _ in range(k)]
        
        for j in range(k):
            
            image_name = 'augmented_' +str(j) + '_' + file_name + '.jpg'
            xml_name = 'augmented_' +str(j) + '_' + file_name + '.xml'
            image_path = os.path.join(dst_path, image_name)
            augmented_xml_path = os.path.join(dst_path, xml_name)
            
            cv2.imwrite(image_path,augmented_list[j][0])
            
            export_to_xml(image_name, W, H, augmented_list[j][1], augmented_xml_path)
            
            #print(f'exported {i}/{k}')
        
    print('Augment successful!')