from pathlib import Path
import pytesseract as pt
import numpy as np
import cv2
from PIL import Image


def preprocess(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    kernel = np.ones((2, 2))
    dilated = cv2.dilate(opening, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded


class OCRModel:
    def recognize_text(self, image: Path) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """

        image = np.array(Image.open(image))
        preprocessed_image = preprocess(image)

        custom_config = r'-l chi_sim+eng --psm 3 --oem 1'
        text = pt.image_to_string(preprocessed_image, config=custom_config)
        return text
