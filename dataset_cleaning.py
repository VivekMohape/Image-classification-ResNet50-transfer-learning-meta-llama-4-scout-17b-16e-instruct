import os
import random
import shutil
from PIL import Image

source_dir = "Dataset"
target_dir = "clean_subset_dataset"

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.makedirs(target_dir)

VALID_EXTENSIONS = ('.jpg','.jpeg','.png','.webp')

for class_name in os.listdir(source_dir):

    class_path = os.path.join(source_dir,class_name)

    if not os.path.isdir(class_path):
        continue

    valid_images=[]

    for img in os.listdir(class_path):

        img_path=os.path.join(class_path,img)

        if not img.lower().endswith(VALID_EXTENSIONS):
            continue

        try:
            with Image.open(img_path) as im:
                im.verify()

            valid_images.append(img)

        except:
            continue

    random.shuffle(valid_images)

    selected_images = valid_images[:500]

    target_class=os.path.join(target_dir,class_name)
    os.makedirs(target_class)

    for img in selected_images:

        shutil.copy(
            os.path.join(class_path,img),
            os.path.join(target_class,img)
        )

print("Dataset cleaned successfully")
