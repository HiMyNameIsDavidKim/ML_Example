from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import requests

LENNA = './data/Lenna.png'
SOCCER = 'https://docs.opencv.org/4.x/roi.jpg'
BUILDING = 'https://amroamroamro.github.io/mexopencv/opencv_contrib/fast_hough_transform_demo_01.png'
GIRL = './data/girl.jpg'
GIRL_WITH_MOM = './data/girl_with_mom.jpg'
HAAR = './data/haarcascade_frontalface_alt.xml'


class LennaService(object):
    def __init__(self):
        pass

    def process(self):
        pass

    def ImageToNumberArray(self, url):
        headers = {'User-Agent': 'My User Agent 1.0'}
        res = requests.get(url, headers=headers)
        img = Image.open(BytesIO(res.content))
        return np.array(img)

    def ExecuteLambda(self, *params):
        cmd = params[0]
        target = params[1]
        if cmd == 'IMAGE_READ':
            return (lambda x: cv2.imread(x, cv2.IMREAD_COLOR))(target)
        elif cmd == 'GRAY_SCALE':
            return (lambda x: x[:, :, 0] * 0.114 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.229)(target)
        elif cmd == 'ARRAY_TO_IMAGE':
            return (lambda x: (Image.fromarray(x)))(target)
        elif cmd == '':
            pass

    def Hough_Line(self, src):
        edges = cv2.Canny(src, 50, 40)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 120, minLineLength=50, maxLineGap=500)
        dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for i in range(lines.shape[0]):
                pt1 = (lines[i][0][0], lines[i][0][1])
                pt2 = (lines[i][0][2], lines[i][0][3])
                cv2.line(dst, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
        return edges, dst

    def Haar_Line(self, src):
        haar = cv2.CascadeClassifier(HAAR)
        dst = src.copy()
        face = haar.detectMultiScale(dst, minSize=(150, 150))
        for (x, y, w, h) in face:
            print(f'얼굴의 좌표 : {x},{y},{w},{h}')
            red = (255, 0, 0)
            cv2.rectangle(dst, (x, y), (x + w, y + h), red, thickness=20)
        return dst, (x, y, w, h)

    def Mosaic_img(self, img, size):
        haar = cv2.CascadeClassifier(HAAR)
        dst = img.copy()
        face = haar.detectMultiScale(dst, minSize=(150, 150))
        for (x, y, w, h) in face:
            print(f'얼굴의 좌표 : {x},{y},{w},{h}')
            (x1, y1, x2, y2) = (x, y, (x + w), (y + h))
            i_rect = img[y1:y2, x1:x2]
            i_small = cv2.resize(i_rect, (size, size))
            i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)
            dst[y1:y2, x1:x2] = i_mos
        return dst

    def menu_1(self, *params):
        print(params[0])
        img = self.ExecuteLambda('IMAGE_READ', params[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f' Shape is {img.shape}')
        plt.imshow('Original', img)
        plt.show()

    def menu_2(self, *params):
        print(params[0])
        src = self.ExecuteLambda('IMAGE_READ', params[1])
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = self.ExecuteLambda('GRAY_SCALE', src)
        plt.imshow('Grey', src)
        plt.show()

    def menu_3(self, *params):
        print(params[0])
        img = self.ExecuteLambda('IMAGE_READ', params[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f'img type : {type(img)}')
        edges = cv2.Canny(np.array(img), 100, 200)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge'), plt.xticks([]), plt.yticks([])
        plt.show()

    def menu_4(self, *params):
        print(params[0])
        img = self.ImageToNumberArray(params[1])
        edges, dst = self.Hough_Line(img)
        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(edges, cmap='gray')
        plt.title('Edge'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(dst, cmap='gray')
        plt.title('Hough'), plt.xticks([]), plt.yticks([])
        plt.show()

    def menu_5(self, *params):
        print(params[0])
        girl = self.ExecuteLambda('IMAGE_READ', params[2])
        girl = cv2.cvtColor(girl, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(girl, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(np.array(girl), 50, 40)
        edgess, dst = self.Hough_Line(girl)
        dst2, rect = self.Haar_Line(girl)
        plt.subplot(231), plt.imshow(girl, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(232), plt.imshow(gray, cmap='gray')
        plt.title('Gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(233), plt.imshow(edges, cmap='gray')
        plt.title('Edges'), plt.xticks([]), plt.yticks([])
        plt.subplot(234), plt.imshow(dst, cmap='gray')
        plt.title('Hough'), plt.xticks([]), plt.yticks([])
        plt.subplot(235), plt.imshow(dst2, cmap='gray')
        plt.title('Haar'), plt.xticks([]), plt.yticks([])
        plt.show()

    def menu_6(self, *params):
        print(params[0])
        girl = self.ExecuteLambda('IMAGE_READ', params[2])
        girl = cv2.cvtColor(girl, cv2.COLOR_BGR2RGB)
        mos = self.Mosaic_img(girl, 10)
        plt.subplot(121), plt.imshow(girl, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(mos, cmap='gray')
        plt.title('Mosaic Image'), plt.xticks([]), plt.yticks([])
        plt.show()


lenna_menus = ['종료',  # 0
               '원본 보기',  # 1
               '그레이 스케일',  # 2
               '엣지 검출',  # 3
               '직선 검출',  # 4
               '얼굴 검출',  # 5
               '소녀 사진 모자이크',  # 6
               '모녀 사진 모자이크',  # 7
               ]

lenna_lambda = {
    "1": lambda t: t.menu_1(lenna_menus[1], LENNA),
    "2": lambda t: api.menu_2(lenna_menus[2], LENNA),
    "3": lambda t: api.menu_3(lenna_menus[3], LENNA),
    "4": lambda t: api.menu_4(lenna_menus[4], BUILDING),
    "5": lambda t: api.menu_5(lenna_menus[5], HAAR, GIRL),
    "6": lambda t: api.menu_6(lenna_menus[6], HAAR, GIRL),
    "7": lambda t: api.menu_6(lenna_menus[7], HAAR, GIRL_WITH_MOM),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    api = LennaService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(lenna_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                lenna_lambda[menu](api)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
