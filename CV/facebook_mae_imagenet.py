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
BATCH_SIZE = 64
NUM_EPOCHS = 8
NUM_WORKERS = 2
LEARNING_RATE = 1.25e-03
pre_model_path = './data/mae_checkpoint/mae_finetuned_vit_base.pth'
fine_model_path = f'./save/mae_vit_base_patch16_i2012_ep{NUM_EPOCHS}_lr{LEARNING_RATE}.pt'
dynamic_model_path = f'./save/mae_vit_base_patch16_i2012_ep'


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
train_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_train)
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class FineTunner(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses = [0]

    def process(self, load=False):
        self.build_model(load)
        # self.finetune_model()
        # self.save_model()

    def build_model(self, load):
        self.model = MaskedAutoencoderViT(patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          decoder_embed_dim=512,
                                          decoder_depth=8,
                                          decoder_num_heads=16,
                                          mlp_ratio=4,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          ).to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = SGD(self.model.parameters(), lr=0)
        if load:
            checkpoint = torch.load(pre_model_path)
            self.model.load_state_dict(checkpoint['model'])
            # self.epochs = checkpoint['epochs']
            # self.losses = checkpoint['losses']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []

    def finetune_model(self):
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            saving_loss = 0.0
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
                if i % 100 == 99:
                    print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                if i % 1000 == 999:
                    self.epochs.append(epoch + 1)
                    self.model = model
                    self.optimizer = optimizer
                    self.losses.append(saving_loss/1000)
                    self.save_model()
                    saving_loss = 0.0
            scheduler.step()
        print('****** Finished Fine-tuning ******')
        self.model = model

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        torch.save(checkpoint, fine_model_path)
#         torch.save(checkpoint, dynamic_model_path+str(self.epochs[-1])+f'_lr{LEARNING_RATE}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = FineTunner()
    trainer.process(load=True)
