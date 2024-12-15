import os
import xml.etree.ElementTree as ET
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import torch

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.classes = self.load_classes(os.path.join(root, "class.txt"))


    def load_classes(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        xml_path = os.path.join(self.root, "annotations", self.images[idx].replace('.jpg', '.xml'))
        boxes, labels = self.parse_xml(xml_path)

        if self.transforms:
            img = self.transforms(img)

        return img, {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.iter('object'):
            label = obj.find('name').text
            if label in self.classes:
                labels.append(self.classes.index(label) + 1)  # Index +1 for 1-based class indexing
                bndbox = obj.find('bndbox')
                boxes.append([
                    float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text),
                ])

        return boxes, labels

# Example of using the dataset
root = 'dataset'  # Path to the dataset
transforms = T.Compose([
    T.ToTensor(),
])

dataset = CustomDataset(root=root, transforms=transforms)

# Print number of images
print(f'Total images: {len(dataset)}')
