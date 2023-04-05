import os
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

Path = os.path.join("./data")
if os.path.exists(Path):
    shutil.rmtree(Path)
# creating three directories: train, val, test
os.makedirs('./data/train/images')
os.makedirs('./data/train/labels')
os.makedirs('./data/val/images')
os.makedirs('./data/val/labels')
os.makedirs('./data/test/images')
os.makedirs('./data/test/labels')


# extractint directory of all images in the folder: ./unsplitted_dataset/images
images = [f for f in os.listdir('./unsplitted_dataset/images') if os.path.isfile(os.path.join('./unsplitted_dataset/images', f))]
labels = [f for f in os.listdir('./unsplitted_dataset/labels') if os.path.isfile(os.path.join('./unsplitted_dataset/labels', f))]
if len(images) != len(labels):
    print("The number of images and labels are not equal")
    exit()


# splitting the dataset into train, val and test in one go
train_images, remaining_images, train_labels, remaining_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
print(f'training_set size: {len(train_images)},         Number of remaining images: {len(remaining_images)}')
val_images, test_images, val_labels, test_labels = train_test_split(remaining_images, remaining_labels, test_size=0.5, random_state=42)
print(f'val_set size: {len(val_images)}')
print(f'test_set size: {len(test_images)}')

# copying images and labels to the corresponding directories
for image in train_images:
    os.system('cp unsplitted_dataset/images/{} data/train/images/'.format(image))
for label in train_labels:
    os.system('cp unsplitted_dataset/labels/{} data/train/labels/'.format(label))

for image in val_images:
    os.system('cp unsplitted_dataset/images/{} data/val/images/'.format(image))
for label in val_labels:
    os.system('cp unsplitted_dataset/labels/{} data/val/labels/'.format(label))

for image in test_images:
    os.system('cp unsplitted_dataset/images/{} data/test/images/'.format(image))
for label in test_labels:
    os.system('cp unsplitted_dataset/labels/{} data/test/labels/'.format(label))


#############################################
####### K-fold Cross Validator method #######
#############################################

# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# for i, (train_index, test_index) in enumerate(kfold.split(images)):
#     print(f"TRAIN: {train_index} TEST: {test_index}")
    
#     # creating train and test directories
#     os.makedirs(f'./data/fold_{i}/train/images')
#     os.makedirs(f'./data/fold_{i}/train/labels')
#     os.makedirs(f'./data/fold_{i}/val/images')
#     os.makedirs(f'./data/fold_{i}/val/labels')
    
#     # copying training data
#     for j in train_index:
#         shutil.copy(f'./unsplitted_dataset/images/{images[j]}', f'./data/fold_{i}/train/images/')
#         shutil.copy(f'./unsplitted_dataset/labels/{labels[j]}', f'./data/fold_{i}/train/labels/')
    
#     for j in test_index:
#         shutil.copy(f'./unsplitted_dataset/images/{images[j]}', f'./data/fold_{i}/val/images/')
#         shutil.copy(f'./unsplitted_dataset/labels/{labels[j]}', f'./data/fold_{i}/val/labels/')