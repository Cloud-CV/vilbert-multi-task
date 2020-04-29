from django.http import JsonResponse
from channels import Group
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .sender import vilbert_task
from .utils import log_to_terminal
from .models import Tasks, QuestionAnswer

import uuid
import os
import random
import traceback
import demo.constants as constants

COCO_PARTIAL_IMAGE_NAME = constants.COCO_PARTIAL_IMAGE_NAME

@csrf_exempt
def vilbert_multitask(request, template_name="index.html"):
    socketid = uuid.uuid4()
    if request.method == "POST":
        try:
            # Fetch the parameters from client side
            socketid = request.POST.get("socket_id")
            task_id = request.POST.get("task_id")
            input_question = request.POST.get("question").lower()
            input_images_list = request.POST.getlist("image_list[]")
            print(input_images_list, input_question, task_id)
            abs_image_path = []
            for i in range(len(input_images_list)):
                abs_image_path.append(str(os.path.join(settings.BASE_DIR, str(input_images_list[i][1:]))))
            print(socketid, task_id, input_question, abs_image_path)
            # Run the Model wrapper
            log_to_terminal(socketid, {"terminal": "Starting Vilbert Multitask Job..."})
            vilbert_task(abs_image_path, str(input_question), task_id, socketid)
        except Exception as e:
            log_to_terminal(socketid, {"terminal": traceback.print_exc()})
    demo_images, images_name = get_demo_images(constants.COCO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images,
                                           "socketid": socketid,
                                           "images_name": images_name})


def get_task_details(request, task_id):
    try:
        task = Tasks.objects.get(unique_id=task_id)
    except Tasks.DoesNotExist:
        response_data = {
            "error": "Tasks with id {} doesn't exist".format(task_id)
        }
        return JsonResponse(response_data)
    response_data = {
        "unique_id": task.unique_id,
        "name": task.name,
        "placeholder": task.placeholder,
        "description": task.description,
        "num_of_images": task.num_of_images,
        "example": task.example
    }
    return JsonResponse(response_data)


def get_demo_images(demo_images_path):
    try:
        image_count = 0
        demo_images = []
        while(image_count<6):
            random_image = random.choice(os.listdir(demo_images_path))
            if COCO_PARTIAL_IMAGE_NAME in random_image:
                demo_images.append(random_image)
                image_count += 1

        demo_images_path = [os.path.join(constants.COCO_IMAGES_URL, x) for x in demo_images]
        images_name = [x for x in demo_images]
    except Exception as e:
        print(traceback.print_exc())
        images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg',]
        demo_images_path = [os.path.join(settings.STATIC_URL, 'images', x) for x in images]
        images_name = [x for x in images]
    return demo_images_path, images_name


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def file_upload(request):
    if request.method == "POST":
        images = request.FILES.getlist("files[]")
        print("Image", images)
        socketid = request.POST.get('socketid')
        dir_type = constants.VILBERT_MULTITASK_CONFIG['image_dir']
        file_paths = []
        for i in images:
            image_uuid = uuid.uuid4()
            image_extension = str(i).split(".")[-1]
            img_path = os.path.join(dir_type, str(image_uuid)) + "." + image_extension
            # handle image upload
            handle_uploaded_file(i, img_path)
            file_paths.append(img_path.replace(settings.BASE_DIR, ""))

        img_url = img_path.replace(settings.BASE_DIR, "")
        return JsonResponse({"file_paths": file_paths})
