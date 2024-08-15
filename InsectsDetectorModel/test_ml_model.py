import coremltools as ct
import numpy as np
import PIL.Image


# Scenario 1: load an image from disk.
def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img


if __name__ == "__main__":
    # Load a model whose input type is "Image".
    model = ct.models.MLModel('models/InsectsDetectorModel.mlpackage')

    Height = 224  # use the correct input image height
    Width = 224  # use the correct input image width

    # Load the image and resize using PIL utilities.
    _, img = load_image('InsectsDetectorModel/sample_data/ant_sample.jpg', resize_to=(Width, Height))
    out_dict = model.predict({'input_1': img})

    print(out_dict)
