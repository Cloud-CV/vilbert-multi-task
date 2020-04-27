from django.conf import settings
from .utils import log_to_terminal

import os
import pika
import sys
import json


def vilbert_task(image_path, question, task_id, socket_id):

    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost',
        port=5672,
        socket_timeout=10000))
    channel = connection.channel()
    queue = "vilbert_multitask_queue"
    channel.queue_declare(queue=queue, durable=True)
    message = {
        'image_path': image_path,
        'question': question,
        'socket_id': socket_id,
        "task_id": task_id
    }
    log_to_terminal(socket_id, {"terminal": "Publishing job to ViLBERT Queue"})
    channel.basic_publish(exchange='',
                          routing_key=queue,
                          body=json.dumps(message),
                          properties=pika.BasicProperties(
                          delivery_mode = 2, # make message persistent
                      ))

    print(" [x] Sent %r" % message)
    log_to_terminal(socket_id, {"terminal": "Job published successfully"})
    connection.close()