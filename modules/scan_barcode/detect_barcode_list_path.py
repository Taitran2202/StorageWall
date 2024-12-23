'''
docker exec docker_barcode python3.7 /home/greystone/LongVu/Projects/StorageWall/Sources/Gitlab_mlserver/ml-sub-storage-wall/modules/scan_barcode/detect_barcode_list_path.py \
   /home/greystone/StorageWall/image/scan_barcode/pathListImage.txt \
      /home/greystone/StorageWall/image/scan_barcode/debug
'''
import json
import os
import sys
import time

import cv2
import numpy as np
from dbr import *

# start_counting time
start_time = time.time()

# you can replace the following variables' value with yours.
license_key = "t0076xQAAADuZblMHxQg58GXQutgTiiKN/ATlr3XybdoANCzVqllT5N3owQ98Jp/F9lBip5ws9HLwlnspGgLq73cMIuwW1MzuWPcHJEgpoA=="
#license_server = "Input the name/IP of the license server"
json_file = r"/home/greystone/Downloads/2/1.txt"

reader = BarcodeReader()

reader.init_license(license_key)
#reader.init_license_from_server(license_server, license_key)
#license_content = reader.output_license_to_string()
#reader.init_license_from_license_content(license_key, license_content)

# error = reader.init_runtime_settings_with_file(json_file)
# if error[0] != EnumErrorCode.DBR_OK:
#     print(error[1])

def checkAndConvertBinary(text):
    text_new = str(text)
    text_new = text_new.replace("\\xd","x")
    text_new = text_new.replace("x8","_")
    return text_new
def calcBlurryImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacianImage = cv2.Laplacian(gray, cv2.CV_64F)
    mean, stddev = cv2.meanStdDev(laplacianImage)
    variance = int(stddev[0][0] ** 2)
    return variance

PATH_IMAGE = sys.argv[1] #os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
print(PATH_IMAGE)
PATH_RESULT = sys.argv[2] #os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
print(PATH_RESULT)
with open(PATH_IMAGE) as fp:  
	line = fp.readline().strip()
	cnt = 1
	jsondata = {}
	while line:
		jsonBarcode = []
		pathImage =line
		print("Line {}: {}".format(cnt, pathImage.strip()))
		try:
			for i in range(4):
				i += 1
				text_results = reader.decode_file(pathImage)
				if text_results != None:
					print(text_results)
					break
				else:
					parent_dir = os.path.dirname(pathImage)
					image_origin = cv2.imread(pathImage)
					image_file_path_cur = os.path.join(parent_dir,f'image_scan_barcode_{i}.jpg')
					cv2.imwrite(image_file_path_cur,image_origin)
				if i == 1:
					print("retry_1_times")
					image_origin = cv2.imread(image_file_path_cur)
					alpha = 2   # Contrast control
					beta = 0    # Brightness control (0-100)
					image = image_origin
					#image = image_origin[0:3120,700:3500,:]
					variance_blur = calcBlurryImage(image)
					cv2.putText(image, "variance_blur: " + str(variance_blur), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					cv2.putText(image, "alpha: " + str(alpha), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					cv2.putText(image, "beta: " + str(beta), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					img_output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
					cv2.imwrite(pathImage, img_output)
				elif i == 2:
					print("retry_2_times")
					image_origin = cv2.imread(image_file_path_cur)
					alpha = 5
					beta = 0
					image = image_origin
					#image = image_origin[0:3120,700:3500,:]
					variance_blur = calcBlurryImage(image)
					cv2.putText(image, "variance_blur: " + str(variance_blur), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					cv2.putText(image, "alpha: " + str(alpha), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					cv2.putText(image, "beta: " + str(beta), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					img_output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
					cv2.imwrite(pathImage, img_output)
				elif i == 3:
					print("retry_3_times")
					image_origin = cv2.imread(image_file_path_cur)
					alpha = 1
					beta = 0
					image = image_origin
					#image = image_origin[0:3120,700:3500,:]
					variance_blur = calcBlurryImage(image)
					cv2.putText(image, "variance_blur: " + str(variance_blur), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
					cv2.putText(image, "alpha: " + str(alpha), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
					cv2.putText(image, "beta: " + str(beta), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
					img_output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
					img_yuv = cv2.cvtColor(img_output, cv2.COLOR_BGR2YUV)
					img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
					img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
					cv2.imwrite(pathImage, img_output)

			if text_results != None:
				for text_result in text_results:
					# print("Barcode Format : ")
					# print(text_result.barcode_format_string)
					# print("Barcode Text : ")
					# print(text_result.barcode_text)
					# print("Localization Points : ")
					# print(text_result.localization_result.localization_points)
					# jsonBarcodeItem = {}
					# jsonBarcodeItem["format"] = text_result.barcode_format_string
					# jsonBarcodeItem["text"] = text_result.barcode_text
					# jsonBarcodeItem["localization"] = text_result.localization_result.localization_points
					# jsonBarcode.append(jsonBarcodeItem)
					print("done")
					print("Barcode Format : ")
					print(text_result.barcode_format_string)
					print("Barcode Text : ")
					print(text_result.barcode_text)
					print("Localization Points : ")
					print(text_result.localization_result.localization_points)

					string  = str(text_result.barcode_bytes)
					string = string.replace("bytearray(b'","")
					string = string.replace("')","")
					print("string : ", string)
					string = checkAndConvertBinary(string)
					print("string : ", string)
					if("Character set invalid" in text_result.barcode_text):
						text_result.barcode_text = string
						print("get from binary")
						print("barcode_text : ", text_result.barcode_text)

					jsonBarcodeItem = {}
					jsonBarcodeItem["format"] = text_result.barcode_format_string
					jsonBarcodeItem["text"] = text_result.barcode_text
					jsonBarcodeItem["localization"] = text_result.localization_result.localization_points
					jsonBarcode.append(jsonBarcodeItem)
		except BarcodeReaderError as bre:
			print(bre)
		jsondata[pathImage] = jsonBarcode
		if line.strip() != '':
			line = fp.readline().strip()
			cnt += 1
	with open( PATH_RESULT, 'w') as outfile:
		json.dump(jsondata, outfile)

# end_time
end_time = time.time() - start_time
print('Elapse time=',end_time)

if __name__ == '__main__':
    image_path = ""
    