import load_data

params = {
    "batch_size": 64,
    "num_workers": 4,
    "n_epochs": 10,
    "image_size": 256,
    "in_channels": 3,
    "num_classes": 5
}

if __name__ == "__main__":
    loaders = load_data.create_datasets()
