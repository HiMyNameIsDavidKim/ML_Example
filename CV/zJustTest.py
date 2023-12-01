import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torch.optim import Adam

data_dir = 'path/to/coco/data'
annotations_dir = 'path/to/coco/annotations'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])
coco_train_dataset = CocoDetection(root=data_dir, annFile=f'{annotations_dir}/instances_train2017.json', transform=transform)
coco_val_dataset = CocoDetection(root=data_dir, annFile=f'{annotations_dir}/instances_val2017.json', transform=transform)
train_data_loader = DataLoader(coco_train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_data_loader = DataLoader(coco_val_dataset, batch_size=2, shuffle=False, num_workers=4)

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 91
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for val_images, val_targets in val_data_loader:
            val_images = list(image.to(device) for image in val_images)
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            val_outputs = model(val_images)

torch.save(model.state_dict(), 'trained_model.pth')
