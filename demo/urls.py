
from django.conf.urls import url

from .views import vilbert_multitask, get_task_details, file_upload

app_name = "demo"
urlpatterns = [
    url(r'^upload_image/', file_upload, name='upload_image'),
    url(r"^get_task_details/(?P<task_id>[0-9]+)/$", get_task_details, name="get_task_details"),
    url(r"^$", vilbert_multitask, name="vilbert_multitask"),
]
