from pathlib import Path
import easyocr
from PIL import Image, JpegImagePlugin
import numpy as np
import cv2

def convert_to_300_dpi(image: JpegImagePlugin):

    """
        This method takes an image file as input converts its DPI to 300.

        :param image: The image object.
    """

    # Set the DPI to 300
    img = img.convert('RGB')  # Convert to RGB mode if the image is not in this mode already
    img = img.resize((int(img.width * 300 / img.info['dpi'][0]), int(img.height * 300 / img.info['dpi'][1])), resample=Image.LANCZOS)
    img.info['dpi'] = (300, 300)
    return img


class OCRModel:
    def recognize_text(self, image: Path) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        
        reader = easyocr.Reader(['ch_sim','en'], gpu = True) 

        im = Image.open(image)
        im = np.array(convert_to_300_dpi(im))


        # Converting the image to grey-scale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Blurring it to reduce noise
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        #inverting the image
        invert = 255 - thresh
        im = invert



        result = reader.readtext(im)

        text = []
        for det in result:
            text.append(det[1])

        return ' '.join(text)