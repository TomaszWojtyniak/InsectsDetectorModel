import glob
from pandas.core.common import flatten
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

params = {
    "batch_size": 64,
    "num_workers": 4,
    "n_epochs": 10,
    "image_size": 256,
    "in_channels": 3,
    "num_classes": 5
}

train_data_path = 'images/train'
test_data_path = 'images/test'
valid_data_path = 'images/valid'
class_to_idx = []
idx_to_class = []


class InsectsDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        return image, label


def visualize_augmentations(image_paths, dataset, idx=0, samples=10, cols=5, random_img=False):
    dataset = copy.deepcopy(dataset)
    rows = samples // cols

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1, len(image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()


def create_datasets():
    train_image_paths = []  # to store image paths in list
    valid_image_paths = []
    test_image_paths = []
    classes = []  # to store class values
    global class_to_idx
    global idx_to_class

    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('/')[-1])
        train_image_paths.append(glob.glob(data_path + '/*'))

    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    for data_path in glob.glob(valid_data_path + '/*'):
        valid_image_paths.append(glob.glob(data_path + '/*'))

    valid_image_paths = list(flatten(valid_image_paths))
    random.shuffle(valid_image_paths)

    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))
    random.shuffle(test_image_paths)

    print("\nTrain size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths),
                                                                   len(test_image_paths)))

    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    train_dataset = InsectsDataset(train_image_paths)
    valid_dataset = InsectsDataset(valid_image_paths)
    test_dataset = InsectsDataset(test_image_paths)

    #visualize_augmentations(train_image_paths, train_dataset, np.random.randint(1, len(train_image_paths)), random_img=True)

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=params["batch_size"], shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=params["batch_size"], shuffle=False
    )

    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    return loaders
