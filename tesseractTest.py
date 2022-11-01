#importing modules
import pytesseract
from PIL import Image


img = Image.open(r"Collection_1/001.png")
cropped_img = img.crop((6, 13, 1628, 1129))
cropped_img = cropped_img.convert('RGB')
cropped_img.save("cropped_img.jpg")
print(pytesseract.image_to_string(cropped_img))
