import load_data
import train_model

if __name__ == "__main__":
    loaders = load_data.create_datasets()
    train_model.train_model(loaders)
