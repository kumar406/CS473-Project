#importing modules
import easyocr
from PIL import Image

def read_image(input):
  reader = easyocr.Reader(['en'])
  output = []
  fileName = input['img']
  img = Image.open(fileName)
  print(fileName)
  for y in input['labels']:
      tArray = [y['label']['name']]
      temp = [0, 0, 0, 0]
      temp[0] = y['box'][1] #xmin
      temp[1] = y['box'][0] #ymin
      temp[2] = y['box'][3] #xmax
      temp[3] = y['box'][2] #ymax
      cropped_img = img.crop(temp)
      cropped_img = cropped_img.convert('RGB')
      cropped_img.save('cropped_img.jpg')
      tArray += reader.readtext('cropped_img.jpg', detail = 0)
      #print(tArray)
      output.append(tArray)
      tArray = []
  print(output)
  return output