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
        r"[$letters]{1}[$numbers]{3}[$letters]{2}[$numbers]{2,3}"
    ).substitute(letters=_LETTERS, numbers=_NUMBERS)
    _LICENSE_PLATE_NUMBER_CASCADE = "haarcascade_russian_plate_number.xml"
    _MODEL_PATH = os.path.join(settings.BASE_DIR, "recognition/model.tflite")
    _SCALE_FACTORS = [1.2, 1.1, 1.3, 1.4, 1.5, 1.6]

    def __init__(self) -> None:
        self._haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + self._LICENSE_PLATE_NUMBER_CASCADE
        )
        self._model = tf.lite.Interpreter(self._MODEL_PATH)

    def get_license_plate_number(self, image_body: bytes) -> str:
        with tempfile.NamedTemporaryFile() as f:
            f.write(image_body)
            for scale_factor in self._SCALE_FACTORS:
                image = self._prepare_image(f.name, scale_factor)

                prediction = self._get_model_prediction(image)
                print(f"Prediction: {prediction}")
                validated_prediction = self._validate(prediction)

                if validated_prediction:
                    return validated_prediction

            return ""

    def _validate(self, license_plate_number: str) -> str:
        # check whether license plate number format is correct
        match = re.match(self._LICENSE_PLATE_NUMBER_PATTERN, license_plate_number)
        return match.group() if match else ""

    def _get_model_prediction(self, image: cv2.typing.MatLike) -> str:
        self._model.allocate_tensors()

        model_input = self._model.get_input_details()
        self._model.set_tensor(model_input[0]["index"], image)

        self._model.invoke()

        model_output = self._model.get_output_details()

        net_out_value = self._model.get_tensor(model_output[0]["index"])

        self._model.reset_all_variables()

        return self._decode_batch(net_out_value)

    def _prepare_image(self, image_path: str, scale_factor: float) -> cv2.typing.MatLike:
        img = cv2.imread(image_path)
        img = self._extract_license_plate_image(img, scale_factor)
        img = self._preprocess_image(img)
        return img

    def _extract_license_plate_image(
        self, image: cv2.typing.MatLike, scale_factor: float
    ) -> cv2.typing.MatLike:

        license_plate_number_rects = self._haar_cascade.detectMultiScale(
            image, scaleFactor=scale_factor, minNeighbors=5
        )
        # detectMultiScale returns empty tuple if it could not recognise license plate number on image
        # it returns ndarray with success
        # comparing by None or not is incorrect, so it raises error
        if license_plate_number_rects is ():
            # could not recognise license plate number on image by cascade
            # so try to return not cropped image
            return image

        x, y, w, h = license_plate_number_rects[0]
        return image[y : y + h, x : x + w]

    def _decode_batch(self, out: np.ndarray) -> str:
        result = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            out_str = "".join(self._CHARS[c] for c in out_best if c < len(self._CHARS))
            result.append(out_str)
        return result[0]

    @staticmethod
    def _preprocess_image(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img.astype(np.float32) / 255
        img = img.T
        img = np.float32(img.reshape(1, 128, 64, 1))
        return img
