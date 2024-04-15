import os
import shutil
import random
from sklearn.model_selection import train_test_split


def split_dataset(input_dir, output_dir, test_size=0.2, valid_size=0.2, random_state=42):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    valid_dir = os.path.join(output_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Split dataset into train and test sets
    train_files, test_valid_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

    # Split remaining files into test and validation sets
    test_files, valid_files = train_test_split(test_valid_files, test_size=valid_size / (1 - test_size),
                                               random_state=random_state)

    # Move files to respective directories
    move_files(train_files, input_dir, train_dir)
    move_files(test_files, input_dir, test_dir)
    move_files(valid_files, input_dir, valid_dir)

    print("Dataset split successfully!")


def move_files(files, source_dir, dest_dir):
    for file in files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(source_path, dest_path)


# Example usage:
input_dir = '/Users/tomaszwojtyniak/Desktop/Insects-dataset/train/spider'
output_dir = '/Users/tomaszwojtyniak/Desktop/dataset-divided'
split_dataset(input_dir, output_dir)
