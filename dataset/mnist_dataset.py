import glob # used to iterate through the directory
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch import Tensor
from typing import Tuple


class MnistDataset(Dataset):
    """
        Custom Pytorch Dataset for MNIST
    """
    def __init__(self, root_dir: str, image_extension: bool="png", train: bool=True) -> None:
        """
            Initialize dataset class.
        """
        split = "train" if train else "test"
        self.image_extension = image_extension
        self.images, self.labels = self.load_images(os.path.join(root_dir, split))
    
    def load_images(self, path: str):
        """
            Search through the data and load all images.
        """
        images = []
        labels = []
        class_directories = os.listdir(path) # [0, 1, 2, 3, ..]
        for class_directory in class_directories:

            # find all files based on their extension
            search_query = glob.glob(os.path.join(path, class_directory, f"*.{self.image_extension}"))
            for filename in search_query:
                images.append(filename)
                labels.append(int(class_directory))

        print(f"Found {len(images)} images")

        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image = Image.open(self.images[index])
        label = self.labels[index]
        

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image)

        return image

