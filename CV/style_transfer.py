import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.utils.data as data
import torchvision.models as models
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

content_layer_num = 1
image_size = 512
epoch = 5000
content_dir = "./data/content/Neckarfront_origin.jpg"
style_dir = "./data/style/monet.jpg"
device = 'mps'


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self, x):
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out


class StyleTransferModel(object):
    def __init__(self):
        self.content = None
        self.style = None
        self.generated = None
        self.cnt = 0

    def process(self, cp):
        global checkpoint, checkpoint_img
        checkpoint = cp
        checkpoint_img = f'./save/style_transfer/{checkpoint}.jpg'
        self.cnt = cp
        self.prepare_img()
        self.modeling()

    def image_preprocess(self, img_dir):
        img = Image.open(img_dir)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                 std=[1, 1, 1]),
        ])
        img = transform(img).view((-1, 3, image_size, image_size))
        return img

    def image_postprocess(self, tensor):
        transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                         std=[1, 1, 1])
        img = transform(tensor.clone())
        img = img.clamp(0, 1)
        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        return img

    def prepare_img(self):
        content = self.image_preprocess(content_dir).to(device)
        style = self.image_preprocess(style_dir).to(device)
        generated = self.image_preprocess(checkpoint_img).to(device)
        generated = generated.clone().requires_grad_().to(device)

        # print(content.requires_grad, style.requires_grad, generated.requires_grad)
        # plt.imshow(self.image_postprocess(content[0].cpu()))
        # plt.show()
        # plt.imshow(self.image_postprocess(style[0].cpu()))
        # plt.show()
        # gen_img = self.image_postprocess(generated[0].cpu()).data.numpy()
        # plt.imshow(gen_img)
        # plt.show()

        self.content = content
        self.style = style
        self.generated = generated

    def modeling(self):
        resnet = Resnet().to(device)
        style_target = [GramMatrix().to(device)(i) for i in resnet(self.style)]
        content_target = resnet(self.content)[content_layer_num]
        style_weight = [1 / n ** 2 for n in [64, 64, 256, 512, 1024, 2048]]

        optimizer = optim.LBFGS([self.generated])

        iteration = [0]
        while iteration[0] < epoch:
            def closure():
                optimizer.zero_grad()
                out = resnet(self.generated)

                style_loss = [GramMSELoss().to(device)(out[i], style_target[i]) * style_weight[i] for i in
                              range(len(style_target))]

                content_loss = nn.MSELoss().to(device)(out[content_layer_num], content_target)

                # style : content = 1000 : 1
                total_loss = 1000 * sum(style_loss) + torch.sum(content_loss)
                total_loss.backward(retain_graph=True)

                if iteration[0] % 100 == 0:
                    print(total_loss)
                    self.save_checkpoint(self.cnt)
                    self.cnt += 1

                iteration[0] += 1
                return total_loss

            optimizer.step(closure)

    def save_checkpoint(self, cnt):
        gen_img = self.image_postprocess(self.generated[0].cpu()).data.numpy()
        gen_img = Image.fromarray((gen_img * 255).astype(np.uint8))
        gen_img.save(f'./save/style_transfer/{cnt}.jpg')

        plt.figure(figsize=(10, 10))
        plt.imshow(gen_img)
        plt.show()

    def show_result(self, arg):
        gen_img = Image.open(f'./save/style_transfer/{arg}.jpg')

        plt.figure(figsize=(10, 10))
        plt.imshow(gen_img)
        plt.show()


style_menus = ["Exit",  # 0
               "Modeling",  # 1
               "Check Image",  # 2
               ]

style_lambda = {
    "1": lambda t: t.process(int(input('Please input start point : '))),
    "2": lambda t: t.show_result(int(input('Please input file name : '))),
    "3": lambda t: print(" ** No Function ** "),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    st = StyleTransferModel()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(style_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                style_lambda[menu](st)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
