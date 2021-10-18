import string
import random
from captcha.image import WheezyCaptcha
from PIL import  Image
chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
model = WheezyCaptcha(chars)


img = model.generate_image('abcd')
img.save('a.png','png')