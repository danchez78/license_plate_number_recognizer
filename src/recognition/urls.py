from django.urls import path

from . import views

app_name = "recognition"
urlpatterns = [
    path("image", views.Prediction.as_view(), name="prediction"),
]
