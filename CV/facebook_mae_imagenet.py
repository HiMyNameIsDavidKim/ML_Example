import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import AdamW, SGD
from torch import nn
from torch.utils.data import random_split
from tqdm import tqdm, tqdm_notebook
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial

import facebook_vit
from mae_util import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from CV.facebook_mae import MaskedAutoencoderViT

gpu_ids = []
device_names = []
if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        gpu_ids += [gpu_id]
        device_names += [torch.cuda.get_device_name(gpu_id)]
print(gpu_ids)
print(device_names)

if len(gpu_ids) > 1:
    gpu = 'cuda:' + str(gpu_ids[0])  # GPU Number
else:
    gpu = "cuda" if torch.cuda.is_available() else "cpu"


device = gpu
BATCH_SIZE = 64  # 1024
NUM_EPOCHS = 100  # 100
WARMUP_EPOCHS = 5  # 5
NUM_WORKERS = 2
LEARNING_RATE = 6.25e-05  # 1e-03
pre_model_path = './save/MAE/mae_finetuned_vit_base_given.pth'
fine_model_path = f'./save/mae_vit_base_i2012_ep{NUM_EPOCHS}_lr{LEARNING_RATE}.pt'
dynamic_model_path = f'./save/mae_vit_base_i2012_ep'


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_set = torchvision.datasets.ImageFolder('../../YJ/ILSVRC2012/train', transform=transform_train)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('../../YJ/ILSVRC2012/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class FineTunner(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses = [0]
        self.accuracies = [0]

    def process(self, load=False):
        self.build_model(load)
        self.finetune_model()
        self.save_model()

    def build_model(self, load):
        self.model = facebook_vit.__dict__['vit_base_patch16'](
            num_classes=1000,
            drop_path_rate=0.1,
            )
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = SGD(self.model.parameters(), lr=0)

        if load:
            checkpoint = torch.load(pre_model_path)
            checkpoint_model = checkpoint['model']
            state_dict = self.model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            interpolate_pos_embed(self.model, checkpoint_model)
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
            trunc_normal_(self.model.head.weight, std=2e-5)
            self.model.to(device)

            if 'given' not in str(pre_model_path):
                self.epochs = checkpoint['epochs']
                self.losses = checkpoint['losses']
                self.accuracies = checkpoint['accuracies']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []
            self.accuracies = []

    def finetune_model(self):
        model = self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            if epoch in range(WARMUP_EPOCHS):
                lr_warmup = ((epoch + 1) / WARMUP_EPOCHS) * LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_warmup
            running_loss = 0.0
            saving_loss = 0.0
            correct = 0
            total = 0
            for i, data in tqdm_notebook(enumerate(train_loader, 0), total=len(train_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                saving_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 100 == 99:
                    print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}, acc: {correct/total*100:.2f} %')
                    running_loss = 0.0
                if i % 1000 == 999:
                    self.model = model
                    self.optimizer = optimizer
                    self.epochs.append(epoch + 1)
                    self.losses.append(saving_loss/1000)
                    self.accuracies.append(correct/total*100)
                    self.save_model()
                    saving_loss = 0.0
                    correct = 0
                    total = 0
            scheduler.step()
        print('****** Finished Fine-tuning ******')
        self.model = model

    def save_model(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epochs': self.epochs,
            'losses': self.losses,
            'accuracies': self.accuracies,
        }
        torch.save(checkpoint, fine_model_path)
#         torch.save(checkpoint, dynamic_model_path+str(self.epochs[-1])+f'_lr{LEARNING_RATE}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = FineTunner()
    trainer.process(load=True)
