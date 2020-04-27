from django.contrib import admin

from .models import Tasks, QuestionAnswer
# from import_export.admin import ImportExportMixin


class ImportExportTimeStampedAdmin(admin.ModelAdmin):
    exclude = ("created_at", "modified_at")


@admin.register(Tasks)
class TaskAdmin(ImportExportTimeStampedAdmin):
    readonly_fields = ("created_at",)
    list_display = (
        "unique_id",
        "name",
        "placeholder",
        "example",
        "num_of_images",
        "description",
    )


@admin.register(QuestionAnswer)
class QuestionAnswerAdmin(ImportExportTimeStampedAdmin):
    readonly_fields = ("created_at",)
    list_display = (
        "task",
        "input_text",
        "input_images",
        "answer_text",
        "answer_images",
        "socket_id",
    )
