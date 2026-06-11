
from django.db import migrations
def create_groups(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    for name in ["Admin", "Analyst", "Viewer"]:
        Group.objects.get_or_create(name=name)
def delete_groups(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    Group.objects.filter(name__in=["Admin", "Analyst", "Viewer"]).delete()
class Migration(migrations.Migration):
    dependencies = [
        ("auth", "__first__"),
    ]
    operations = [migrations.RunPython(create_groups, delete_groups)]
