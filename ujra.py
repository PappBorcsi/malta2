import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageDraw
import numpy as np
import base64
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from engine import train_one_epoch, evaluate
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import utils
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os

class CustomFastRCNNPredictor(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomFastRCNNPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class CustomMaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels, hidden_layer, num_classes):
        super(CustomMaskRCNNPredictor, self).__init__()
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_layer, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layer, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.mask_head(x)


# képek és annotációk betöltésére
class CustomDataset(Dataset):
    def __init__(self, annotation_file, transforms=None):
        self.transforms = transforms
        self.annotation_file = annotation_file

        with open(annotation_file) as f:
            self.data = json.load(f)

        img_data = base64.b64decode(self.data['imageData'])
        self.img = Image.open(BytesIO(img_data)).convert("RGB")

        self.img_path = self.data['imagePath']
        self.shapes = self.data['shapes']

        self.annotations = self._parse_annotations()

    def _parse_annotations(self):
        annotations = []
        for shape in self.shapes:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'])
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                bbox = [x_min, y_min, x_max, y_max]
                annotations.append({
                    'bbox': bbox,
                    'label': shape['label'],
                    'points': points  # Tároljuk el a pontokat a maszkokhoz
                })
        return annotations

    def _create_mask(self, width, height, annotations):
        mask = Image.new('L', (width, height), 0)  # Létrehoz egy üres, fekete képet
        draw = ImageDraw.Draw(mask)

        # Rajzoljuk meg a maszkot minden annotációhoz
        for ann in annotations:
            polygon = [(x, y) for x, y in ann['points']]
            draw.polygon(polygon, outline=1, fill=1)

        return torch.as_tensor(np.array(mask), dtype=torch.uint8)  # Maszk tensor

    def __getitem__(self, idx):
        img = self.img
        annotations = self.annotations

        width, height = img.size

        boxes = [ann['bbox'] for ann in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.tensor([0] * len(boxes), dtype=torch.int64)  # Minden box osztálya 0

        masks = self._create_mask(width, height, annotations).unsqueeze(0)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])  # Add hozzá az image_id-t
        }

        transform = T.ToTensor()
        img = transform(img)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return 1

# Kép megjelenítése dobozokkal
def show_image_with_boxes(img, target):
    # Ha img Tensor, alakítsd át NumPy tömbbé
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()  # Átalakítás

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    if 'boxes' in target:
        boxes = target['boxes'].numpy()
        for box in boxes:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
    ax.set_title("picture with box")
    plt.show()

    # Load Mask R-CNN model
def load_model(model_path, num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Update the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Update the mask
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels  # Access correct mask layer
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)

    # Load the saved model state
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


    # visssza adja képhez tartozó predikciókat
def test_model(model, img_path):
    # Kép betöltése és előkészítése
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    return prediction


def custom_collate_fn(batch):
    return tuple(zip(*batch))

def main():
    model_path = 'maskrcnn_finetuned.pth'
    annotation_file = r"C:\Users\Papp Borcsi\Downloads\images\3_mouse.json"
    img_path = r"C:\Users\Papp Borcsi\Downloads\images\3_mouse.jpg"

    dataset = CustomDataset(annotation_file=annotation_file)
    img, target = dataset[0]
    show_image_with_boxes(img, target)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")

        # If the file does not exist, train the model
        print("Training starts...")

        """dataset = CustomDataset(annotation_file=annotation_file)
        data_loader= DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)
        """
        data_loader = DataLoader(
                   dataset, # adathalmaz
                   batch_size=4,  # egyszerre 2 mintát (képet és annotációit) fog betölteni és visszaadni
                   shuffle=False, #adatokat véletlen adja viss
                   num_workers=0,
                   collate_fn=custom_collate_fn  # saját funkciód, amely a batch-hez tartozó minták megfelelő formátumra alakításáért felelős.
        )

        print("data_loader")
        print(data_loader)
        data_iter = iter(data_loader)
        images, targets = next(data_iter)

        # Nézd meg az első kép és az első target tartalmát
        print(f"pictures size: {images[0].shape}")
        print(f"Target: {targets[0]}")

        for i, (images, targets) in enumerate(data_loader):
            print(f"Batch {i + 1}:")
            print(f"pictures size: {images[0].shape}")  # Az első kép mérete az első batch-ből
            print(f"Target: {targets[0]}")  # Az első target az első batch-ből

        images, targets = next(iter(data_loader))

        # Válassz ki egy képet és a hozzá tartozó annotációkat
        image = images[0].permute(1, 2, 0).numpy()  # Átrendezés HWC (height, width, channel) formára
        target = targets[0]  # Az első target (bounding boxok, maszkok stb.)

        print("2.picture")
        # Jelenítsd meg a képet
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Bounding boxok megjelenítése
        if 'boxes' in target:
            boxes = target['boxes'].numpy()
            for box in boxes:
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.show()
        print("end............")
        """
                # Display image and label.
                dataloadertest = next(iter(data_loader))
                print(f"Labels batch shape: {dataloadertest.size()}")
                img = dataloadertest[0].squeeze()
                label = dataloadertest[0]
                plt.imshow(img, cmap="gray")
                plt.show()
                print(f"Label: {label}")
        """

        print("Creating model...")
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes=2)
        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        print("3.picture")
        dataloadertest = next(iter(data_loader))
        images, targets = dataloadertest
        img = images[0].squeeze()  # Az első image tensorból eltávolítjuk a dimenziókat, ha szükséges
        target = targets[0]  # Az első target

        # Create a figure and axis
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0).numpy())  # Ha RGB, akkor a tengelyeket átrendezzük (H, W, C)

        # Bounding boxok megjelenítése
        if 'boxes' in target:
            boxes = target['boxes'].numpy()
            for box in boxes:
                rect = patches.Rectangle(
                    (box[0], box[1]),  # (x_min, y_min)
                    box[2] - box[0],  # width (x_max - x_min)
                    box[3] - box[1],  # height (y_max - y_min)
                    linewidth=2,  # Line thickness
                    edgecolor='r',  # Red color for the rectangle
                    facecolor='none'  # No fill color
                )
                ax.add_patch(rect)  # Add the rectangle to the plot
        plt.title("Image with Bounding Boxes")
        plt.show()


        print("Training created...")

        num_epochs = 10
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            print("stop1...")
            evaluate(model, data_loader, device)
            print("stop2...")

        print("Saving the model...")

        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    else:
        print(f"Model file found: {model_path}")

    # Load and test the model
    print("Loading the model...")
    model = load_model(model_path, num_classes=2)
    model.to(device)

    print("Testing image...")
    prediction = test_model(model, img_path)

    # Checking predicted classes
    print("Predicted classes:", prediction[0]['labels'])
    print("Prediction scores:", prediction[0]['scores'])
    print("Prediction boxes:", prediction[0]['boxes'])
    if 'masks' in prediction[0]:
        print("Prediction mask shape:", prediction[0]['masks'].shape)

if __name__ == '__main__':
    main()
