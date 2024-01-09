import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL
from torch.optim import AdamW, SGD
from torch import nn
from torch.utils.data import random_split
from tqdm import tqdm, tqdm_notebook
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial

import facebook_vit
from mae_util import interpolate_pos_embed
from timm.models.layers import trunc_normal_

import facebook_mae
import mae_misc as misc
from mae_misc import NativeScalerWithGradNormCount as NativeScaler
from CV.util.visualization import inverse_transform, inout_images_plot


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
BATCH_SIZE = 64  # 1024 // 64
NUM_EPOCHS = 100  # 100 // 800
WARMUP_EPOCHS = 5  # 5 // 40
NUM_WORKERS = 2
LEARNING_RATE = 3.125e-05  # paper: 1e-03 // 1.5e-04 -> implementation: 5e-04 // 1.5e-04
pre_model_path = f'./save/mae_base_i2012_ep{NUM_EPOCHS}_lr{LEARNING_RATE}.pt'
load_model_path = './save/MAE/mae_finetuned_vit_base_given.pth'
fine_model_path = f'./save/mae_vit_base_i2012_ep{NUM_EPOCHS}_lr{LEARNING_RATE}.pt'
dynamic_model_path = f'./save/mae_vit_base_i2012_ep'


transform_train = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_train)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class FineTuner(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
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
            global_pool=True,
            )
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = SGD(self.model.parameters(), lr=0)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS)

        if load:
            checkpoint = torch.load(load_model_path, map_location=device)
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

            if 'given' not in str(load_model_path):
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
            if epoch < WARMUP_EPOCHS:
                lr_warmup = ((epoch + 1) / WARMUP_EPOCHS) * LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_warmup
                if epoch + 1 == WARMUP_EPOCHS:
                    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            print(f"epoch {epoch + 1} learning rate : {optimizer.param_groups[0]['lr']}")
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
                    self.epochs.append(epoch + 1)
                    self.losses.append(saving_loss/1000)
                    self.accuracies.append(correct/total*100)
                    saving_loss = 0.0
                    correct = 0
                    total = 0
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.save_model()
            scheduler.step()
        print('****** Finished Fine-tuning ******')

    def save_model(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epochs': self.epochs,
            'losses': self.losses,
            'accuracies': self.accuracies,
        }
        torch.save(checkpoint, fine_model_path)
#         torch.save(checkpoint, dynamic_model_path+str(self.epochs[-1])+f'_lr{LEARNING_RATE}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


class PreTrainer(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = [0]
        self.losses = [0]
        self.accuracies = [0]

    def process(self, load=False):
        self.build_model(load)
        self.pretrain_model()
        self.save_model()

    def build_model(self, load):
        self.model = facebook_mae.__dict__['mae_vit_base_patch16_dec512d8b'](norm_pix_loss=True).to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = SGD(self.model.parameters(), lr=0)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS)

        if load:
            checkpoint = torch.load(load_model_path, map_location=device)
            msg = self.model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
            if 'given' not in str(load_model_path):
                self.epochs = checkpoint['epochs']
                self.losses = checkpoint['losses']
                self.accuracies = checkpoint['accuracies']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []
            self.accuracies = []

    def pretrain_model(self):
        model = self.model.train()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        loss_scaler = NativeScaler()

        for epoch in range(NUM_EPOCHS):
            if epoch < WARMUP_EPOCHS:
                lr_warmup = ((epoch + 1) / WARMUP_EPOCHS) * LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_warmup
                if epoch + 1 == WARMUP_EPOCHS:
                    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            print(f"epoch {epoch + 1} learning rate : {optimizer.param_groups[0]['lr']}")
            running_loss = 0.0
            saving_loss = 0.0
            for i, data in tqdm_notebook(enumerate(train_loader, 0), total=len(train_loader)):
                samples, _ = data
                samples = samples.to(device, non_blocking=True)

                optimizer.zero_grad()

                loss, pred, mask = model(samples, mask_ratio=.75)
                loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
                # Scaler include loss.backward() and optimizer.step()

                running_loss += loss.item()
                saving_loss += loss.item()

                if i % 100 == 99:
                    print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                    inout_images_plot(samples=samples, mask=mask, pred=pred, model=model)
                if i % 1000 == 999:
                    self.epochs.append(epoch + 1)
                    self.losses.append(saving_loss / 1000)
                    saving_loss = 0.0
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.save_model()
            scheduler.step()
        print('****** Finished Fine-tuning ******')

    def save_model(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epochs': self.epochs,
            'losses': self.losses,
        }
        torch.save(checkpoint, pre_model_path)
        #         if int(self.epochs[-1]) % 10 == 9:
        #             torch.save(checkpoint, dynamic_model_path+str(self.epochs[-1])+f'_lr{LEARNING_RATE}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process(load=False)
