'''
python3 modules/detect_phone_slot/script_augment_labelme.py \
    dataset/model_detect_phone_slot/images/data2/train 300
'''
import os
import cv2
import sys
import json
import base64
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

# definition
def image_to_base64(image):
    # Convert the image to a byte array
    _, buffer = cv2.imencode('.jpg', image)
    
    # Encode the byte array to base64
    encoded_image = base64.b64encode(buffer)

    return encoded_image.decode("utf-8")

def make_polys(json_file):
    # read json file
    with open(json_file, "r") as js:
        json_data = json.load(js)
    # create empty list of polys
    polys = [] # list of tuple
    # append poly shape to list of polys
    for shape in json_data['shapes']:
        # This assert might be overkill but better safe that sorry ...
        assert shape['shape_type'] == "polygon"
        polys.append(Polygon(shape['points'], label=shape['label']))
    # get image shape
    img_shape = (json_data['imageHeight'], json_data['imageWidth'], 3)
    # convert to
    polys_oi = PolygonsOnImage(polys, shape=img_shape)
    return(polys_oi)

def check_dimension(x,y,W,H):
    if x < 0:
        x = 0
    if x > W:
        x = W
    if y < 0:
        y = 0
    if y > H:
        y = H
    return [x,y]

def make_annotation(image_name,img,polys):
    # init empty dict
    json_content = {}
    
    # add key-value pairs
    json_content["version"] = "4.5.6"
    json_content["flags"] = {}
    json_content["imagePath"]=image_name
    json_content["imageData"] = image_to_base64(img)
    json_content["imageHeight"] = img.shape[0]
    json_content["imageWidth"] = img.shape[1]
    
    # add shapes
    shapes = []
    for poly in polys:
        # init empty dict of shape
        current_shape = {}
        #add key-value pairs
        current_shape["group_id"]=None
        current_shape["shape_type"]="polygon"
        current_shape["flags"]={}
        current_shape["label"]=poly.label
        # current_shape["points"] = [ [list(coord)[0].astype('float'), list(coord)[1].astype('float')] for coord in poly.coords]
        current_shape["points"] = [ check_dimension(list(coord)[0].astype('float'),list(coord)[1].astype('float'),img.shape[1],img.shape[0]) for coord in poly.coords]
        shapes.append(current_shape)

    json_content["shapes"] = shapes

    return json_content

# initialize
M = int(sys.argv[2])
image_files  = []
json_files = []
image_tails = ["jpg","png","jpeg"]
annotation_dir = sys.argv[1]
src_path = sys.argv[1]
dst_path = 'dst'

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
    iaa.Rotate((-3,3))
    ], random_order=True) # apply augmenters in random order

if __name__ == "__main__":
    
    print('Import successful!')

    # Create debug path
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    
    # Check files
    files = os.listdir(annotation_dir)
    
    # check image file
    for file in files:
        file_name = file.split(".")[0]
        file_tail = file.split(".")[-1]
        if file_tail in image_tails:
            image_files.append(file)
            json_files.append(file_name + '.json')
    
    # Calculate image_augmented per image
    N = len(image_files)
    k = int(M/N)

    for i in range(N):
        print(f'({i+1}/{N}) augment image={image_files[i]} json_file={json_files[i]}')
        
        file_name = image_files[i].split(".")[0] 
        image_path = os.path.join(src_path, image_files[i])
        json_path = os.path.join(src_path, json_files[i]) 

        print(image_path)

        # read image, json file
        img = cv2.imread(image_path)

        # make poly_oi of current json file
        polys_oi = make_polys(json_path)

        # create list of augmented
        augmented_list = [my_augmenter(image = img, polygons = polys_oi) for _ in range(k)]

        for i,(img_augmented, polys_augmented) in enumerate(augmented_list):
            image_name = file_name + '_' + str(i) + '.jpg'
            image_augmented_path = os.path.join(dst_path, image_name)
            # export image
            cv2.imwrite(image_augmented_path, img_augmented)
            # export poly to labelme format
            current_json_content = make_annotation(image_name, img_augmented, polys_augmented)
            with open(os.path.join(dst_path,image_name.replace('jpg','json')),'w') as json_file:
                json.dump(current_json_content, json_file, indent=2)
                
    print('Augment successful!')