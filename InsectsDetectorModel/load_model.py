import pathlib
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from torchinfo import summary
from pathlib import Path
import os
import coremltools as ct
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision
from torchviz import make_dot


class CNN(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 53 * 53,
                      out_features=output_shape)
        )

    def forward(self, x):
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


class ImageFolderCustom(Dataset):
    # 2. Initialize our custom dataset
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        # 3. Create class attributes
        # Get all of the image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        # Setup transform
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Create a function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite __len__()
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite __getitem__() method to return a particular sample
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.name  # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return untransformed image and label

    def get_class_ids(self) -> torch.Tensor:
        """Returns a list of class IDs for every image in the dataset."""
        class_ids = [self.class_to_idx[path.parent.name] for path in self.paths]
        return torch.tensor(class_ids)

    def get_class_names(self) -> List[str]:
        "Returns a list of class names for every image in the dataset."
        class_names = [path.parent.name for path in self.paths]
        return class_names


def convert_model(loaded_model, class_labels: List[str]):
    # Set the model in evaluation mode.
    loaded_model.eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(loaded_model, example_input)
    out = traced_model(example_input)

    scale = 1 / (0.226 * 255.0)
    bias = [- 0.485 / (0.229), - 0.456 / (0.224), - 0.406 / (0.225)]

    image_input = ct.ImageType(name="input_1",
                               shape=example_input.shape,
                               scale=scale, bias=bias)

    model = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
        classifier_config=ct.ClassifierConfig(class_labels),
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17
    )

    # Save the converted model.
    model.save("models/InsectsDetectorModel.mlpackage")


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file structure.")

    # 3. Create a dictionary of index labels (computers prefer numbers rather than strings as labels)
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None):
    """Makes a prediction on a target image with a trained model and plots the image and prediction."""
    # Load in the image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # Transform if necessary
    if transform:
        target_image = transform(target_image)

    # Turn on eval/inference mode and make a prediction
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image (this is the batch dimension, e.g. our model will predict on batches of 1x image)
        target_image = target_image.unsqueeze(0)

        # Make a prediction on the image with an extra dimension
        target_image_pred = model(target_image)  # make sure the target image is on the right device

    # Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert predction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0))  # remove batch dimension and rearrange shape to be HWC
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show();


if __name__ == "__main__":
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)

    MODEL_NAME = "Insects_detector_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    image_path = Path("InsectsDetectorModel/images")

    if image_path.is_dir():
        print(f"{image_path} directory exists")
    else:
        print(f"{image_path} directory does not exist")

    test_dir = image_path / "test"

    loaded_model = CNN(input_shape=3,
                       hidden_units=100,
                       output_shape=14)

    # Load in the save state_dict()
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    summary(loaded_model, input_size=[32, 3, 224, 224])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])

    custom_image_transform = transforms.Compose([
        transforms.Resize(size=(224, 224))
    ])

    test_data = ImageFolderCustom(targ_dir=test_dir,
                                  transform=test_transforms)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=32,
                                 shuffle=False)

    class_names = test_data.classes
    class_to_idx = test_data.class_to_idx

    sample_data_path = Path("InsectsDetectorModel/sample_data")
    custom_image_path = sample_data_path / "mantis_sample.jpg"

    # pred_and_plot_image(model=loaded_model,
    #                     image_path=custom_image_path,
    #                     class_names=class_names,
    #                     transform=custom_image_transform)

    # y_preds = []
    # loaded_model.eval()
    # with torch.inference_mode():
    #     for X, y in tqdm(test_dataloader, desc="Making predictions"):
    #         # Do the forward pass
    #         y_logit = loaded_model(X)
    #         # Turn predictions from logits -> prediction probabilities -> predictions labels
    #         y_pred = torch.softmax(y_logit, dim=1).argmax(
    #             dim=1)  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    #         # Put predictions on CPU for evaluation
    #         y_preds.append(y_pred.cpu())
    # # Concatenate list of predictions into a tensor
    # y_pred_tensor = torch.cat(y_preds)
    #
    # confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    # confmat_tensor = confmat(preds=y_pred_tensor,
    #                          target=test_data.get_class_ids())
    #
    # # 3. Plot the confusion matrix
    # fig, ax = plot_confusion_matrix(
    #     conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
    #     class_names=class_names,  # turn the row and column labels into class names
    #     figsize=(10, 7)
    # )
    # plt.show();

    convert_model(loaded_model, class_labels=class_names)
