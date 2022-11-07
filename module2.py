#importing modules
import easyocr
import identify_image
from PIL import Image

LABEL_MAP_PATH = 'Tensorflow\\workspace\\annotations\\label_map.pbtxt'
IMAGE_PATH = 'Tensorflow\\workspace\\images\\test\\110.png'
CHECKPOINT_PATH = 'Tensorflow\\workspace\\models\\bingus_model\\ckpt-4'
PIPELINE_CONFIG_PATH = 'Tensorflow\\workspace\\models\\bingus_model\\pipeline.config'

model = identify_image.load_model(PIPELINE_CONFIG_PATH, CHECKPOINT_PATH)

labeled_imgs = identify_image.label_images([IMAGE_PATH], LABEL_MAP_PATH, model)

print(labeled_imgs)

# img = Image.open("Collection_1/001.png")
# cropped_img = img.crop((2674, 1, 2993, 181))
# cropped_img = cropped_img.convert('RGB')
# cropped_img.save("cropped_img.jpg")
# reader = easyocr.Reader(['en'])
# print(reader.readtext('Collection_1/001.png', detail = 0))

#rint(pytesseract.image_to_string(cropped_img, lang='eng', config='--psm 6'))

# input list
# img = Image.open(r"input file")
#dict = {"entity": [1, 313, 723, 676], "rel_attr":[757, 1, 1076, 186], "rel":[1001, 330, 1323, 659], "entity":[2935, 310, 3652, 669]}
#arr = []
#for x in dict:
#    temp = []
#    cropped_img = img.crop(dict[x])
#    cropped_img = cropped_img.convert('RGB')
#    temp.append(x)
#    temp.append(pytesseract.image_to_string(cropped_img).strip())
#    arr.append(temp)

#print(arr)