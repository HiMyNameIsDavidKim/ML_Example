import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import seaborn as sns

from CV.util.classname_imagenet import imagenet_ind2str

device = 'mps'
BATCH_SIZE = 1
NUM_WORKERS = 2

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_origin = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
origin_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_origin)
test_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
origin_loader = DataLoader(origin_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 데이터셋 커스텀 transform, 데이터셋 합치기
# class YourDataset2(Dataset):
#     def __init__(self):
#         pass
#
#     def __getitem__(self, idx):
#         img = self.data[idx]
#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Lambda(shuffler),
#             transforms.ToTensor(),
#         ])
#         img = transform(img)
#         label = self.labels[idx]
#         return img, label
#
# concat_dataset = ConcatDataset([dataset1, dataset2])
# concat_dataloader = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def shuffler(img):
    d = 7
    sub_imgs = []
    for i in range(d):
        for j in range(d):
            sub_img = img[i * 224 // d:(i + 1) * 224 // d, j * 224 // d:(j + 1) * 224 // d]
            sub_imgs.append(sub_img)
    np.random.shuffle(sub_imgs)
    new_img = np.vstack([np.hstack([sub_imgs[i] for i in range(d*j, d*(j+1))]) for j in range(d)])
    return new_img

def rotator(img):
    d = 7
    sub_imgs = []
    for i in range(d):
        for j in range(d):
            sub_img = img[i * 224 // d:(i + 1) * 224 // d, j * 224 // d:(j + 1) * 224 // d]
            sub_imgs.append(sub_img)
    sub_imgs = [np.rot90(sub_img) for sub_img in sub_imgs]
    new_img = np.vstack([np.hstack([sub_imgs[i] for i in range(d * j, d * (j + 1))]) for j in range(d)])
    return new_img

def show_img(n, shuffle=False, rotate=False):
    for i, data in enumerate(origin_loader):
        if i == n:
            inputs, labels = data
            inputs_np, labels_np = inputs.numpy(), labels.numpy()
            inputs_np = np.transpose(inputs_np, (0, 2, 3, 1))[0]
            if shuffle:
                inputs_np = shuffler(inputs_np)
            if rotate:
                inputs_np = rotator(inputs_np)
            plt.imshow(inputs_np)
            plt.title(imagenet_ind2str(int(labels_np)))
            plt.show()
            break

def show_reverse_img(images, labels):
    std_array = np.reshape([0.229, 0.224, 0.225], (1, 1, 3))
    mean_array = np.reshape([0.485, 0.456, 0.406], (1, 1, 3))
    reversed_img = images * std_array + mean_array
    plt.imshow(reversed_img)
    plt.title(imagenet_ind2str(int(labels)))
    plt.show()

def cal_conf(n, shuffle=False, rotate=False):
    model = timm.models.vit_base_patch16_224(pretrained=True)
    print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Classes: {model.num_classes}')
    print(f'****** Model Creating Completed. ******')
    model.to(device).eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx == n:
                images = images.numpy()
                images = np.transpose(images, (0, 2, 3, 1))[0]
                if shuffle:
                    images = shuffler(images)
                if rotate:
                    images = rotator(images)

                show_reverse_img(images, labels)

                images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images = images.to(device)
                labels = labels.to(device)
                outputs, extr = model(images)

                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                conf = probs[int(labels)].to('cpu')

                print(f'Label : {imagenet_ind2str(int(labels))}')
                print(f'Predict : {imagenet_ind2str(int(pred))}')
                print(f'Confidence of label : {float(conf):.3f}')
                print(extr.shape)
                break

def cal_dist(tensor1, tensor2):
    squared_diff = np.square(tensor1 - tensor2)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance


class PatchHeatMapViT(object):
    def __init__(self):
        self.model = None
        self.model_cnn = None
        self.img_origin = None
        self.img_trans = None
        self.labels = None
        self.tensor_origin = None
        self.tensor_trans = None
        self.conf_origin = 0
        self.conf_trans = 0
        self.conf_diff = []
        self.list_labels = []

    def process_v(self, n, shuffle=False, rotate=False):
        self.build_model()
        self.extract_tnc(n, shuffle, rotate)
        self.visual()

    def process_g(self, shuffle=False, rotate=False):
        self.build_model()
        self.grouping(shuffle, rotate)

    def process_g_cnn(self, shuffle=False, rotate=False):
        self.build_model_cnn()
        self.grouping_cnn(shuffle, rotate)

    def process_d(self, n, shuffle=False, rotate=False):
        self.build_model()
        self.build_model_cnn()
        self.extract_tnc_compare(n, shuffle, rotate)

    def build_model(self):
        self.model = timm.models.vit_base_patch16_224(pretrained=True)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Classes: {self.model.num_classes}')
        print(f'****** Model Creating Completed. ******\n')

    def build_model_cnn(self):
        self.model_cnn = torch.hub.load('szq0214/MEAL-V2','meal_v2', 'mealv2_resnest50_cutmix', pretrained=True)
        print(f'Parameter: {sum(p.numel() for p in self.model_cnn.parameters() if p.requires_grad)}')
        print(f'****** Model Creating Completed. ******\n')

    def extract_tnc(self, n, shuffle, rotate):
        self.model.to(device).eval()
        with torch.no_grad():
            (images, labels) = test_set[n]
            images = images.reshape(1, 3, 224, 224).float()
            labels = torch.tensor(labels)
            images = images.numpy()
            images = np.transpose(images, (0, 2, 3, 1))[0]
            images_t = images
            if shuffle:
                images_t = shuffler(images_t)
            if rotate:
                images_t = rotator(images_t)

            self.img_origin = images
            self.img_trans = images_t
            self.labels = labels

            images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
            images = images.to(device)
            labels = labels.to(device)
            outputs, self.tensor_origin = self.model(images)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            self.conf_origin = probs[int(labels)].to('cpu')

            images_t = torch.from_numpy(images_t.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
            images_t = images_t.to(device)
            labels = labels.to(device)
            outputs, self.tensor_trans = self.model(images_t)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            self.conf_trans = probs[int(labels)].to('cpu')

    def visual(self):
        show_reverse_img(self.img_origin, self.labels)
        show_reverse_img(self.img_trans, self.labels)
        print(f'Label : {imagenet_ind2str(int(self.labels))}')
        print(f'Confidence of origin : {float(self.conf_origin):.3f}')
        print(f'Confidence of trans : {float(self.conf_trans):.3f}')

        np_origin = self.tensor_origin.to('cpu').numpy()
        np_origin = np_origin.reshape(197, 768)
        np_origin = np.delete(np_origin, 0, axis=0)
        np_trans = self.tensor_trans.to('cpu').numpy()
        np_trans = np_trans.reshape(197, 768)
        np_trans = np.delete(np_trans, 0, axis=0)

        dists = [cal_dist(i, j) for i, j in zip(np_origin, np_trans)]
        df = np.array(dists).reshape(14, 14)

        sns.heatmap(data=df,
                    annot=True,
                    cmap='Oranges',
                    linewidths=.5,
                    vmax=50,
                    vmin=-0,
                    cbar_kws={'shrink': .5})
        plt.show()

    def grouping(self, shuffle, rotate):
        self.model.to(device).eval()
        with torch.no_grad():
            for idx, (images, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                images = images.numpy()
                images = np.transpose(images, (0, 2, 3, 1))[0]
                images_t = images
                if shuffle:
                    images_t = shuffler(images_t)
                if rotate:
                    images_t = rotator(images_t)

                self.img_origin = images
                self.img_trans = images_t
                self.labels = labels

                images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images = images.to(device)
                labels = labels.to(device)
                outputs, __ = self.model(images)
                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                self.conf_origin = probs[int(labels)].to('cpu')

                images_t = torch.from_numpy(images_t.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images_t = images_t.to(device)
                labels = labels.to(device)
                outputs, __ = self.model(images_t)
                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                self.conf_trans = probs[int(labels)].to('cpu')

                self.conf_diff.append(float(self.conf_origin-self.conf_trans))
                self.list_labels.append(int(self.labels))

    def extract_tnc_compare(self, n, shuffle, rotate):
        self.model.to(device).eval()
        self.model_cnn.to(device).eval()
        with torch.no_grad():
            (images, labels) = test_set[n]
            images = images.reshape(1, 3, 224, 224).float()
            labels = torch.tensor(labels)
            images = images.numpy()
            images = np.transpose(images, (0, 2, 3, 1))[0]
            images_t = images
            if shuffle:
                images_t = shuffler(images_t)
            if rotate:
                images_t = rotator(images_t)

            self.img_origin = images
            self.img_trans = images_t
            self.labels = labels

            images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
            images = images.to(device)
            labels = labels.to(device)
            outputs, self.tensor_origin = self.model(images)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            self.conf_origin = probs[int(labels)].to('cpu')

            images_t = torch.from_numpy(images_t.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
            images_t = images_t.to(device)
            labels = labels.to(device)
            outputs, self.tensor_trans = self.model(images_t)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            self.conf_trans = probs[int(labels)].to('cpu')

            print(f'Label : {imagenet_ind2str(int(self.labels))}')
            print(f'Diff of ViT : {float(self.conf_origin)-float(self.conf_trans):.3f}')

            outputs = self.model_cnn(images)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            self.conf_origin = probs[0, int(labels)].to('cpu')

            outputs = self.model_cnn(images_t)
            _, pred = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            self.conf_trans = probs[0, int(labels)].to('cpu')

            print(f'Diff of CNN : {float(self.conf_origin) - float(self.conf_trans):.3f}')

    def grouping_cnn(self, shuffle, rotate):
        self.model_cnn.to(device).eval()
        with torch.no_grad():
            for idx, (images, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                images = images.numpy()
                images = np.transpose(images, (0, 2, 3, 1))[0]
                images_t = images
                if shuffle:
                    images_t = shuffler(images_t)
                if rotate:
                    images_t = rotator(images_t)

                self.img_origin = images
                self.img_trans = images_t
                self.labels = labels

                images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model_cnn(images)
                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                self.conf_origin = probs[0, int(labels)].to('cpu')

                images_t = torch.from_numpy(images_t.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images_t = images_t.to(device)
                labels = labels.to(device)
                outputs = self.model_cnn(images_t)
                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                self.conf_trans = probs[0, int(labels)].to('cpu')

                self.conf_diff.append(float(self.conf_origin-self.conf_trans))
                self.list_labels.append(int(self.labels))


class PatchHeatMapCNN(object):
    def __init__(self):
        self.model = None
        self.img_origin = None
        self.img_trans = None
        self.labels = None
        self.tensor_origin = None
        self.tensor_trans = None
        self.conf_origin = 0
        self.conf_trans = 0
        self.conf_diff = []
        self.list_labels = []
        self.pred_origin = None
        self.pred_trans = None

    def process_v(self, n, shuffle=False, rotate=False):
        self.build_model()
        self.extract_tnc(n, shuffle, rotate)
        self.visual()

    def process_g(self, shuffle=False, rotate=False):
        self.build_model()
        self.grouping(shuffle, rotate)

    def build_model(self):
        self.model = torch.hub.load('szq0214/MEAL-V2','meal_v2', 'mealv2_resnest50_cutmix', pretrained=True)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'****** Model Creating Completed. ******\n')

    def extract_tnc(self, n, shuffle, rotate):
        self.model.to(device).eval()
        with torch.no_grad():
            (images, labels) = test_set[n]
            images = images.reshape(1, 3, 224, 224).float()
            labels = torch.tensor(labels)
            images = images.numpy()
            images = np.transpose(images, (0, 2, 3, 1))[0]
            images_t = images
            if shuffle:
                images_t = shuffler(images_t)
            if rotate:
                images_t = rotator(images_t)

            self.img_origin = images
            self.img_trans = images_t
            self.labels = labels

            images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
            images = images.to(device)
            labels = labels.to(device)
            outputs = self.model(images)
            _, pred = torch.max(outputs, 1)
            self.pred_origin = int(pred.to('cpu'))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            self.conf_origin = probs[0, int(labels)].to('cpu')

            images_t = torch.from_numpy(images_t.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
            images_t = images_t.to(device)
            labels = labels.to(device)
            outputs = self.model(images_t)
            _, pred = torch.max(outputs, 1)
            self.pred_trans = int(pred.to('cpu'))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            self.conf_trans = probs[0, int(labels)].to('cpu')

    def visual(self):
        show_reverse_img(self.img_origin, self.labels)
        show_reverse_img(self.img_trans, self.labels)
        print(f'Label : {imagenet_ind2str(int(self.labels))}')
        print(f'Confidence of origin : {float(self.conf_origin):.3f}')
        print(f'Confidence of trans : {float(self.conf_trans):.3f}')
        print(f'Diff : {float(self.conf_origin) - float(self.conf_trans):.3f}')

    def grouping(self, shuffle, rotate):
        self.model.to(device).eval()
        with torch.no_grad():
            for idx, (images, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                images = images.numpy()
                images = np.transpose(images, (0, 2, 3, 1))[0]
                images_t = images
                if shuffle:
                    images_t = shuffler(images_t)
                if rotate:
                    images_t = rotator(images_t)

                self.img_origin = images
                self.img_trans = images_t
                self.labels = labels

                images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images = images.to(device)
                labels = labels.to(device)
                outputs, __ = self.model(images)
                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                self.conf_origin = probs[int(labels)].to('cpu')

                images_t = torch.from_numpy(images_t.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images_t = images_t.to(device)
                labels = labels.to(device)
                outputs, __ = self.model(images_t)
                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                self.conf_trans = probs[int(labels)].to('cpu')

                self.conf_diff.append(float(self.conf_origin-self.conf_trans))
                self.list_labels.append(int(self.labels))


patchTrans_menus = ["Exit",  # 0
                    "Show Image(Original)",  # 1
                    "Calculate Confidence of Image(Original)",  # 2
                    "Show Image(Shuffle)",  # 3
                    "Calculate Confidence of Image(Shuffle)",  # 4
                    "Show Image(Rotate)",  # 5
                    "Calculate Confidence of Image(Rotate)",  # 6
                    "Heat Map Visualization",  # 7
                    ]

patchTrans_lambda = {
    "1": lambda t: show_img(int(input('Please input image number : '))),
    "2": lambda t: cal_conf(int(input('Please input image number : '))),
    "3": lambda t: show_img(int(input('Please input image number : ')), shuffle=True),
    "4": lambda t: cal_conf(int(input('Please input image number : ')), shuffle=True),
    "5": lambda t: show_img(int(input('Please input image number : ')), rotate=True),
    "6": lambda t: cal_conf(int(input('Please input image number : ')), rotate=True),
    "7": lambda t: t.process_v(int(input('Please input image number : ')), shuffle=True),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    t = PatchHeatMapCNN()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(patchTrans_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                patchTrans_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
