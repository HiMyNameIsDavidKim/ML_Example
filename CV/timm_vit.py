import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

device = 'mps'
BATCH_SIZE = 100
NUM_WORKERS = 2


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class ViTImageNet21k(object):
    def __init__(self):
        self.model = None

    def process(self):
        self.build_modeL()
        self.eval_model()

    def build_modeL(self):
        self.model = timm.models.vit_base_patch16_224(pretrained=True).to(device)
        # self.model = timm.models.vit_large_patch16_224(pretrained=True).to(device)
        print(f'Parameter : {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

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
                print(f"top-1 percentage :  {correct_top1 / total * 100:0.2f}%")
                print(f"top-5 percentage :  {correct_top5 / total * 100:0.2f}%")

        print(f"top-1 percentage :  {correct_top1 / total * 100:0.2f}%")
        print(f"top-5 percentage :  {correct_top5 / total * 100:0.2f}%")


if __name__ == "__main__":
    ViTImageNet21k().process()
