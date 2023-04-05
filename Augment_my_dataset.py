import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import cv2

transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['class_labels']))

# walk through the input_of_augmentation folder and save each file path in a list with only os.
# Then, we will use this list to iterate over the images and apply the transformations.

# get the images_file_names
images_file_names = []
for r, d, f in os.walk('input_of_augmentation/images'):
    for file in f:
        if '.jpg' in file:
            images_file_names.append(file)
print(images_file_names)

# get the input_of_augmentation
labels_file_name = []
for r, d, f in os.walk('input_of_augmentation/labels'):
    for file in f:
        if '.txt' in file:
            labels_file_name.append(file)
print(labels_file_name)

# copy all images inside the images_file_names into a new folder (unsplitted_dataset)
if  os.path.exists('unsplitted_dataset'):
    shutil.rmtree('unsplitted_dataset')
os.mkdir('unsplitted_dataset')
os.mkdir('unsplitted_dataset/images')
os.mkdir('unsplitted_dataset/labels')

for image in images_file_names:
    os.system('cp input_of_augmentation/images/{} unsplitted_dataset/images/'.format(image))
for label in labels_file_name:
    os.system('cp input_of_augmentation/labels/{} unsplitted_dataset/labels/'.format(label))


# define transform in albumentations
transform = A.Compose([
    A.RandomRotate90(always_apply=True, p=1),
    A.ShiftScaleRotate(rotate_limit=30, rotate_method='ellipse', shift_limit=0.0, scale_limit=(-0.1,0), always_apply=True, p=1),
    # A.Transpose (always_apply=True, p=1),
    # A.RandomBrightnessContrast(always_apply=True, p=1),
    # A.Affine (rotate=45, rotate_method="largest_box", fit_output=True, keep_ratio=False, always_apply=True, p=1),
    # A.RandomCrop(width=640, height=640),
    # A.ShiftScaleRotate(shift_limit=0.9, scale_limit=0, rotate_limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, rotate_method='largest_box', always_apply=True, p=1),
    # A.RandomRotate90 (always_apply=True, p=1),
    # A.Rotate (limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=True, p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.6, label_fields=['class_labels']))

# loop through the images and apply the transformations
for i in range(len(images_file_names)):
    # import the image
    image = cv2.imread(os.path.join('input_of_augmentation/images', images_file_names[i]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # import the labels
    labels = np.loadtxt(os.path.join('input_of_augmentation/labels', labels_file_name[i]))

    class_labels  = labels[:, 0]
    bboxes = labels[:, 1:]
    
    for j in range(20):
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = np.array(transformed['bboxes'])
        transformed_class_labels = np.array(transformed['class_labels']).reshape(-1, 1)
        transformed_labels = np.hstack((transformed_class_labels, transformed_bboxes))

        # fig, ax = plt.subplots(1, 2, figsize=(17, 10))
        # ax[0].imshow(image)
        # ax[1].imshow(transformed_image)
        # height, width, channels = transformed_image.shape
        # for k in range(len(transformed_bboxes)):
        #     x_center, y_center, w, h = transformed_bboxes[k][0]*width, transformed_bboxes[k][1]*height, transformed_bboxes[k][2]*width, transformed_bboxes[k][3]*height
        #     rect = patches.Rectangle((x_center-w/2, y_center-h/2), w, h, linewidth=0.7, edgecolor='r', facecolor='none')
        #     ax[1].add_patch(rect)
        # plt.show()

        image_name = images_file_names[i].split('.')[0]
        label_name = labels_file_name[i].split('.')[0]
        image_index = str(j) if j > 9 else '0' + str(j)
        cv2.imwrite(f'unsplitted_dataset/images/{image_name}_{image_index}.jpg', cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
        np.savetxt(f'unsplitted_dataset/labels/{label_name}_{image_index}.txt', transformed_labels)