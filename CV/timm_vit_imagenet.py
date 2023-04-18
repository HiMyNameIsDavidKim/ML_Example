import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from keras.optimizers import Adam
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

device = 'mps'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100
NUM_EPOCHS = 10
NUM_WORKERS = 2
LR = 0.001
model_path = f'./save/ViT_ImageNet21k.pt'

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# train_set = torchvision.datasets.ImageFolder('./data/ImageNet/train', transform=transform_train)
# train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class ViTImageNet21k(object):
    def __init__(self):
        self.optimizer = None
        self.model = None

    def process_eval(self):
        self.build_model()
        self.eval_model()

    def process_finetune(self):
        self.build_model()
        self.finetune_model()
        self.save_model()

    def build_model(self):
        self.model = timm.models.vit_base_patch16_224(pretrained=True).to(device)
        # self.model = timm.models.vit_large_patch16_224(pretrained=True).to(device)
        # self.model = torch.load(model_path).to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Classes: {self.model.num_classes}')

    def eval_model(self):
        model = self.model
        model.to(device).eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct_top1 += (pred == labels).sum().item()

                _, rank5 = outputs.topk(5, 1, True, True)
                rank5 = rank5.t()
                correct = rank5.eq(labels.view(1, -1).expand_as(rank5))
                for k in range(6):
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                correct_top5 += correct_k.item()

                print(f"Step : {idx + 1} / {int(len(test_set) / int(labels.size(0)))}")
                print(f"top-1 Accuracy :  {correct_top1 / total * 100:0.2f}%")
                print(f"top-5 Accuracy :  {correct_top5 / total * 100:0.2f}%")

        print(f"top-1 Accuracy :  {correct_top1 / total * 100:0.2f}%")
        print(f"top-5 Accuracy :  {correct_top5 / total * 100:0.2f}%")

    def finetune_model(self):
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0, last_epoch=-1)
        model.train()
        for epoch in range(3):
            print(f"Epoch {epoch + 1}/3")
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                if i % 50 == 0:
                    with torch.no_grad():
                        model.eval()
                        correct_top1 = 0
                        total = 0
                        for j, (images, labels) in enumerate(test_loader):
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = model(images)

                            _, pred = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct_top1 += (pred == labels).sum().item()

                        print(f"Step : {j + 1} / {int(len(test_set) / int(labels.size(0)))}")
                        print(f"top-1 Accuracy :  {correct_top1 / total * 100:0.2f}%")

                    scheduler.step()

        self.model = model
        self.optimizer = optimizer

    def save_model(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    ViTImageNet21k().process_eval()
    # ViTImageNet21k().process_finetune()