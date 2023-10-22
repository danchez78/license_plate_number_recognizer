from django.urls import include
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("api/", include("recognition.urls")),
    path("admin/", admin.site.urls),
]
