import os
from torchvision.datasets import MNIST # data from pytorch
from tqdm import tqdm # utilized to show a progress bar

def save_mnist_as_pngs(root_dir: str="./data", train: bool=True) -> None:
    """
        Download MNIST dataset and save as individual PNG files.
    """
    # create directory structure
    split = "train" if train else "test"
    save_dir = os.path.join(root_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    
    # download MNIST dataset 
    dataset = MNIST("./", train=train, download=True, transform=None) 
    
    # save each image as PNG
    for idx, (image, label) in enumerate(tqdm(dataset)):
        # create subdirectory for each class (e.g. 0, 1, 2, 3, ...)
        label_dir = os.path.join(save_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        image_path = os.path.join(label_dir, f"{idx:05d}.png")
        image.save(image_path)

if __name__ == "__main__":
    print("Saving training set...")
    save_mnist_as_pngs(train=True)
    
    print("Saving test set...")
    save_mnist_as_pngs(train=False)
    
    print("Done! Dataset saved in mnist_png directory")