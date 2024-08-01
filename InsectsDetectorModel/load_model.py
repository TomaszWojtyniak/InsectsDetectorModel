import torch
from pathlib import Path
from torch import nn
from torchinfo import summary
import coremltools as ct


class TinyVGG(nn.Module):
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


if __name__ == "__main__":
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)

    MODEL_NAME = "Insects_detector_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    loaded_model = TinyVGG(input_shape=3,
                           hidden_units=100,
                           output_shape=14)

    # Load in the save state_dict()
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    summary(loaded_model, input_size=[32, 3, 224, 224])

    #Load model to CoreML

    # Set the model in evaluation mode.
    loaded_model.eval()

    # Trace the model with random data.
    example_input = torch.rand(32, 3, 224, 224)
    traced_model = torch.jit.trace(loaded_model, example_input)
    out = traced_model(example_input)

    # Using image_input in the inputs parameter:
    # Convert to Core ML program using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=example_input.shape)]
    )

    # Save the converted model.
    model.save("models/Insects_detector_model.mlpackage")
