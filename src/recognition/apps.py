import itertools
import numpy as np
import os
import re
import tempfile
from string import Template

import cv2
import tensorflow as tf
from cv2 import data
from django.conf import settings


class Recognizer:
    _NUMBERS = "0123456789"
    _LETTERS = "АВЕКМНОРСТУХ"
    _CHARS = sorted(list(set(_NUMBERS).union(set(_LETTERS))))
    _LICENSE_PLATE_NUMBER_PATTERN = Template(
        r"[$letters]{1}[$numbers]{3}[$letters]{2}[$numbers]{2,3}").substitute(letters=_LETTERS, numbers=_NUMBERS)
    _LICENSE_PLATE_NUMBER_CASCADE = 'haarcascade_russian_plate_number.xml'
    _MODEL_PATH = os.path.join(settings.BASE_DIR, 'recognition/model.tflite')
    _PREDICT_ATTEMPTS = 3

    def __init__(self) -> None:
        self._haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self._LICENSE_PLATE_NUMBER_CASCADE)
        self._model = tf.lite.Interpreter(self._MODEL_PATH)
        self._model_input = self._model.get_input_details()
        self._model_output = self._model.get_output_details()

    def get_license_plate_number(self, image_body: bytes) -> str:
        with tempfile.NamedTemporaryFile() as f:
            f.write(image_body)
            image = self._get_image(f.name)
            for attempt in range(self._PREDICT_ATTEMPTS):
                prediction = self._get_model_prediction(image)
                validated_prediction = self._validate(prediction)
                if not validated_prediction:
                    continue
                return validated_prediction
            return ""

    def _validate(self, license_plate_number: str) -> str:
        match = re.match(self._LICENSE_PLATE_NUMBER_PATTERN, license_plate_number)
        if match:
            return match.group()
        return ""

    def _get_model_prediction(self, image) -> str:
        self._model.allocate_tensors()
        self._model.set_tensor(self._model_input[0]['index'], image)
        self._model.invoke()

        net_out_value = self._model.get_tensor(self._model_output[0]['index'])
        pred_texts = self._decode_batch(net_out_value)
        return pred_texts

    def _get_image(self, image_path: str) -> cv2.typing.MatLike:
        img = cv2.imread(image_path)
        img = self._license_plate_number_extract(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img.astype(np.float32)
        img /= 255
        img = img.T
        img = np.float32(img.reshape(1, 128, 64, 1))
        return img

    def _license_plate_number_extract(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        license_plate_number_rects = self._haar_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

        for x, y, w, h in license_plate_number_rects:
            return image[y+5:y+h-5, x+5:x+w-5]

    def _decode_batch(self, out: np.ndarray) -> str:
        result = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            out_str = ''
            for c in out_best:
                if c < len(self._CHARS):
                    out_str += self._CHARS[c]
            result.append(out_str)
        return result[0]
