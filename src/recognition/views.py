from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .apps import Recognizer


recognizer = Recognizer()


class Prediction(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request: Request) -> Response:
        image_file = request.data["file"]
        try:
            number = recognizer.get_license_plate_number(image_file.read())
        except Exception as exc:
            response_dict = {"error": str(exc)}
            return Response(response_dict, status.HTTP_503_SERVICE_UNAVAILABLE)
        if number:
            response_dict = {"result": number}
            return Response(response_dict, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_204_NO_CONTENT)
