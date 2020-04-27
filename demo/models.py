from django.db import models
from django.utils.html import format_html

class TimeStampedModel(models.Model):
    """
    An abstract base class model that provides self-managed `created_at` and
    `modified_at` fields.
    """

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
        app_label = "demo"


class Tasks(TimeStampedModel):
    unique_id = models.PositiveIntegerField(unique=True)
    name = models.CharField(max_length=1000, blank=True, null=True)
    placeholder = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    num_of_images = models.PositiveIntegerField()
    example = models.CharField(max_length=1000, null=True, blank=True)

    class Meta:
        app_label = "demo"
        db_table = "tasks"


class QuestionAnswer(TimeStampedModel):
    task = models.ForeignKey(Tasks)
    input_text = models.TextField(null=True, blank=True)
    input_images = models.CharField(max_length=10000, null=True, blank=True)
    answer_text = models.TextField(null=True, blank=True)
    answer_images = models.CharField(max_length=10000, null=True, blank=True)
    socket_id = models.CharField(max_length=1000, null=True, blank=True)
    class Meta:
        app_label = "demo"
        db_table = "questionanswer"

    def img_url(self):
        return format_html("<img src='{}' height='150px'>", self.image)

class Attachment(models.Model):
    file = models.FileField(upload_to='attachments')