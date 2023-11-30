import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torch.optim import Adam

# COCO 데이터셋 경로 및 어노테이션 파일 경로 설정
data_dir = 'path/to/coco/data'
annotations_dir = 'path/to/coco/annotations'

# 데이터 전처리 및 변환 설정
transform = transforms.Compose([
    transforms.ToTensor(),
])

# COCO 데이터셋 로드
coco_train_dataset = CocoDetection(root=data_dir, annFile=f'{annotations_dir}/instances_train2017.json', transform=transform)
coco_val_dataset = CocoDetection(root=data_dir, annFile=f'{annotations_dir}/instances_val2017.json', transform=transform)

# DataLoader 설정
train_data_loader = DataLoader(coco_train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_data_loader = DataLoader(coco_val_dataset, batch_size=2, shuffle=False, num_workers=4)

# 모델 생성
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 91  # COCO 데이터셋 클래스 수는 91개

# 클래스 분류기(classifier)를 COCO 클래스 수에 맞게 수정
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

# GPU 사용 가능하면 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)  # 이 부분은 모델에 따라 다르게 수정할 수 있습니다.
        loss.backward()
        optimizer.step()

    # 검증 루프
    model.eval()
    with torch.no_grad():
        for val_images, val_targets in val_data_loader:
            val_images = list(image.to(device) for image in val_images)
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            val_outputs = model(val_images)
            # 검증 결과에 대한 추가적인 처리를 수행할 수 있습니다.

# 학습이 완료된 모델을 저장하려면 아래와 같이 사용할 수 있습니다.
torch.save(model.state_dict(), 'trained_model.pth')
