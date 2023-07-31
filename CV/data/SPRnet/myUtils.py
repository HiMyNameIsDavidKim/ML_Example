import numpy as np
import torch
from torch.nn import functional
from torchvision import transforms
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.feature import canny

device = "cuda" if torch.cuda.is_available() else "cpu"

H_a = np.array([[[1, -1],[0, 0]],
                    [[1, 0], [-1, 0]],
                    [[1, 0], [0, -1]],
                    [[0, 1], [-1, 0]]])

H_b = np.array([[[1, -2, 1], [0, 0, 0], [0, 0, 0]],
                [[1, 0, 0],[-2, 0, 0],[1, 0, 0]],
                [[1, 0, 0], [0, -2, 0], [0, 0, 1]],
                [[0, 0, 1], [0, -2, 0], [1, 0, 0]]])

# bilinear interpolation filter
H_BI=torch.tensor([[1/9, 2/9, 1/3, 2/9, 1/9],
                 [2/9, 4/9, 2/3, 4/9, 2/9],
                 [1/3, 2/3, 1, 2/3, 1/3],
                 [2/9, 4/9, 2/3, 4/9, 2/9],
                 [1/9, 2/9, 1/3, 2/9, 1/9]])

H_BI_2 = torch.tensor([[1/4, 1/2, 1/4],
                        [1/2, 1, 1/2],
                        [1/4, 1/2, 1/4]])

# directional interpolation filter
H_DI=torch.tensor([[1/3, 0, 1/3, 0, 1/3],
                 [0, 2/3, 2/3, 2/3, 0],
                 [1/3, 2/3, 1, 2/3, 1/3],
                 [0, 2/3 , 2/3, 2/3, 0],
                 [1/3, 0, 1/3, 0, 1/3]])

IF_g = torch.tensor([[0.25, 0.50, 0.25],
                    [0.50, 1.0, 0.50],
                    [0.25, 0.50, 0.25]])
IF_b = torch.tensor([[0, 0, 0, 0.06, 0, 0, 0],
                    [0, 0, 0.19, 0.25, 0.19, 0, 0],
                    [0, 0.19, 0.50, 0.56, 0.50, 0.19, 0],
                    [0.06, 0.25, 0.56, 1.0, 0.56, 0.25, 0.06],
                    [0, 0.19, 0.50, 0.56, 0.50, 0.19, 0],
                    [0, 0, 0.19, 0.25, 0.19, 0, 0],
                    [0, 0, 0, 0.06, 0, 0, 0]])
filter_g = IF_g.unsqueeze(0).unsqueeze(0).to(device)
filter_b = IF_b.unsqueeze(0).unsqueeze(0).to(device)

a1 = np.e ** (-2j * np.pi / 3)
a2 = np.e ** (2j * np.pi / 3)

c_rgb2ycbcr = np.array([[0.299, 0.587, 0.114],
               [-0.1687, -0.3313, 0.5],
               [0.5, -0.4187, -0.0813]])
offset_rgb2ycbcr = np.array([16, 128, 128])
mult_rgb2ycbcr= np.array([219, 224, 224])

def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

def tri_circulant(tensor, dim):
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 3).flip((-1,))

def subblock_A(n, H):
    k0 = torch.sum(H**2) * 2 - 1
    k1 = torch.sum(2 * H[:,1] * H[:,4])

    arr = torch.zeros(n)
    arr[0:2] = torch.tensor([k0, k1])
    arr[-1] = k1

    arr_out = circulant(arr, dim=0)

    return arr_out

def subblock_B1(n, H):
    k2 = torch.sum(2 * H[1,:] * H[4,:])
    k3 = 2 * (H[0,0] * H[1,1] + H[0,1] * H[1,0])
    k4 = 2 * (H[0,4] * H[1,3] + H[0,3] * H[1,4])

    arr = torch.zeros(n)
    arr[0:2] = torch.tensor([k2, k4])
    arr[-1] = k3

    arr_out = circulant(arr, dim=0)

    return arr_out

def subblock_B2(n, H):
    k2 = torch.sum(2 * H[1,:] * H[4,:])
    k3 = 2 * (H[0,0] * H[1,1] + H[0,1] * H[1,0])
    k4 = 2 * (H[0,4] * H[1,3] + H[0,3] * H[1,4])

    arr = torch.zeros(n)
    arr[0:2] = torch.tensor([k2, k3])
    arr[-1] = k4

    arr_out = circulant(arr, dim=0)

    return arr_out

def subblock_C(n, H):
    arr = torch.zeros(3 * n)
    arr[0:3] = H[2, 2:]
    arr[-2:] = H[2, :2]

    arr_out = tri_circulant(arr, dim=0)
    
    return arr_out

def subblock_D1(n, H):
    arr = torch.zeros(3 * n)
    arr[0:3] = H[1, 2:]
    arr[-2:] = H[1, :2]

    arr_out = tri_circulant(arr, dim=0)
    
    return arr_out

def subblock_D2(n, H):
    arr = torch.zeros(3 * n)
    arr[0:3] = H[3, 2:]
    arr[-2:] = H[3, :2]

    arr_out = tri_circulant(arr, dim=0)
    
    return arr_out

def subblock_E1(n, H):
    arr = torch.zeros(3 * n)
    arr[0:3] = H[0, 2:]
    arr[-2:] = H[0, :2]

    arr_out = tri_circulant(arr, dim=0)
    
    return arr_out

def subblock_E2(n, H):
    arr = torch.zeros(3 * n)
    arr[0:3] = H[4, 2:]
    arr[-2:] = H[4, :2]

    arr_out = tri_circulant(arr, dim=0)
    
    return arr_out

def NineToOne(sampled_img):
    height, width, channel = sampled_img.shape
    sheight = (int)(height / 3)
    swidth = (int)(width / 3)

    sampled_img = sampled_img[:sheight*3,:swidth*3,:]
    
    out = torch.zeros(sheight, swidth, channel, dtype=torch.float, device=device)
    
    out[:,:,0] = torch.sum(sampled_img[:,:,0].reshape(sheight,3, swidth,3).float(), dim=(1,3))
    out[:,:,1] = torch.sum(sampled_img[:,:,1].reshape(sheight,3, swidth,3).float(), dim=(1,3))
    out[:,:,2] = torch.sum(sampled_img[:,:,2].reshape(sheight,3, swidth,3).float(), dim=(1,3))
    
    return out

def DPD(img):
    height, width, channel = img.shape
    
    # DPD mask
    R_dpd = torch.zeros(height, width, dtype=int, device=device)

    R_dpd[1::3,1::3] = 1
    G_dpd = R_dpd.clone().detach()
    B_dpd = R_dpd.clone().detach()

    tensor_img = torch.tensor(img, device=device)
    img_dpd = torch.zeros(height, width, channel, dtype=float,device=device)

    # apply mask
    img_dpd[:, :, 0] = tensor_img[:, :, 0] * R_dpd
    img_dpd[:, :, 1] = tensor_img[:, :, 1] * G_dpd
    img_dpd[:, :, 2] = tensor_img[:, :, 2] * B_dpd
    
    return img_dpd

def DSD(img):
    height, width, channel = img.shape
    
    # DPD mask
    R_dsd = torch.zeros(height, width, dtype=int, device=device)
    G_dsd = torch.zeros(height, width, dtype=int, device=device)
    B_dsd = torch.zeros(height, width, dtype=int, device=device)

    R_dsd[1::3,0::3] = 1
    G_dsd[1::3,1::3] = 1
    B_dsd[1::3,2::3] = 1

    tensor_img = torch.tensor(img, device=device)
    img_dsd = torch.zeros(height, width, channel, dtype=float,device=device)

    # apply mask
    img_dsd[:, :, 0] = tensor_img[:, :, 0] * R_dsd
    img_dsd[:, :, 1] = tensor_img[:, :, 1] * G_dsd
    img_dsd[:, :, 2] = tensor_img[:, :, 2] * B_dsd
    
    return img_dsd

def DDSD(img):
    height, width, channel = img.shape
    
    # DDSD mask
    R_ddsd = torch.zeros(height, width, dtype=int, device=device)
    G_ddsd = torch.zeros(height, width, dtype=int, device=device)
    B_ddsd = torch.zeros(height, width, dtype=int, device=device)

    R_ddsd[0::3,0::3] = 1
    G_ddsd[1::3,1::3] = 1
    B_ddsd[2::3,2::3] = 1

    tensor_img = torch.tensor(img, device=device)
    img_ddsd = torch.zeros(height, width, channel, dtype=float,device=device)

    # apply mask
    img_ddsd[:, :, 0] = tensor_img[:, :, 0] * R_ddsd
    img_ddsd[:, :, 1] = tensor_img[:, :, 1] * G_ddsd
    img_ddsd[:, :, 2] = tensor_img[:, :, 2] * B_ddsd
    
    return img_ddsd

def myfft(img, storeShift=False, storeFft=False, isColor=True):
    if(isColor): # 채널이 존재하면
        height, width, channel = img.shape
        fft = np.ndarray(shape=channel, dtype=object)
        
        # 2d discrete fourier transform
        for c in range(3):
            fft[c] = np.fft.fft2(img[:,:,c])
    
        # 0, 0을 중심으로 이동
        fft_shift = np.ndarray(shape=channel, dtype=object)
        for c in range(3):
            fft_shift[c] = np.fft.fftshift(fft[c])
    
        # plot 위해 log scale(dB) 적용
        out = np.ndarray(shape=channel, dtype=object)
        for c in range(3):
            out[c] = 20 * np.log(np.abs(fft_shift[c]) + 1) # log 취했을 때 음수 나오는것 방지하기 위해 offset 1 줌
            
    else: # 단일채널(흑백) 이면
        fft = np.fft.fft2(img)
        
        fft_shift = np.fft.fftshift(fft)
        
        out = 20 * np.log(np.abs(fft_shift) + 1)

        
    if(storeShift):
        if(storeFft):
            return out, fft_shift, fft
        else:
            return out, fft_shift
    else:
        if(storeFft):
            return out, fft
        else:
            return out

def myifft(fft_shift, isColor=True):
    if(isColor): # 채널이 존재하면
        channel = fft_shift.shape
        # 역변환
        inverse_shift = np.ndarray(shape=channel, dtype=object)
        inverse_fft = np.ndarray(shape=channel, dtype=object)
        out = np.ndarray(shape=channel, dtype=object)
        for c in range(3):
            inverse_shift[c] = np.fft.ifftshift(fft_shift[c])
            inverse_fft[c] = np.fft.ifft2(inverse_shift[c])
            out[c] = inverse_fft[c].real
    else:
        inverse_shift = np.fft.ifftshift(fft_shift)
        inverse_fft = np.fft.ifft2(inverse_shift)
        out= inverse_fft.real
        
    return out

def myPDAF(img, showProcess=False, storeDPD=False):
    img = np.array(img, dtype=np.float32)
    if(img.max() > 1):
        img = (img / 255 - 1/2) * 2
    else:
        img = (img - 1/2) * 2

    # anti-aliasing filter 적용 위해 fft
    L_out, L_shift, L_fft = myfft(img, storeShift=True, storeFft=True)

    height, width, channel = img.shape
    cheight = (int)(height/2)
    cwidth = (int)(width/2)

    f_cut_vert = 1/6 * height
    f_cut_hori = 1/6 * width

    if(showProcess):
        print(f_cut_vert)
        print(f_cut_hori)

    # low-pass mask
    mask = np.zeros(L_shift[0].shape)
    mask[cheight-(int)(f_cut_vert):cheight+(int)(f_cut_vert) + 1, cwidth-(int)(f_cut_hori):cwidth+(int)(f_cut_hori) + 1] = 1

    L_shift_pdaf = np.ndarray(shape=channel, dtype=object)
    L_shift_pdaf_mag = np.ndarray(shape=channel, dtype=object)
    for c in range(channel):
        L_shift_pdaf[c] = L_shift[c] * mask
        L_shift_pdaf_mag[c] = 20 * np.log(np.abs(L_shift_pdaf[c] + 1))
    
    if(showProcess):
        temp = np.concatenate((L_shift_pdaf_mag[0], L_shift_pdaf_mag[1], L_shift_pdaf_mag[2]), axis=1)
        plt.imshow(temp, cmap='gray')

    L_pdaf = myifft(L_shift_pdaf)

    temp = np.ndarray(shape=img.shape, dtype=float)
    for c in range (3):
        temp[:,:,c] = ((L_pdaf[c]/2 + 1/2) * 255)# .astype(np.int32)
    
    L_pdaf = np.maximum(0, np.minimum(temp, 255))
    L_pdaf_dpd = DPD(L_pdaf)
    S_pdaf_dpd = NineToOne(L_pdaf_dpd).cpu()

    if(storeDPD):
        img_dpd = DPD(img)
        S_dpd = NineToOne(img_dpd).cpu()
        return S_pdaf_dpd, S_dpd
    else:
        return S_pdaf_dpd
    
def PSNR(img_ori, img_comp):
    ref_data = np.array(img_ori, dtype=np.float32)
    target_data = np.array(img_comp, dtype=np.float32)
    
    max_val = 255.0
    mse = np.mean((ref_data - target_data)**2)

    if(mse == 0):
        return -1

    psnr = 10 * np.log10((max_val**2) / mse)
    return psnr

def SSIM(img_ori, img_comp, rgb=True):
    ref_data = np.array(img_ori, dtype=np.float32)
    target_data = np.array(img_comp, dtype=np.float32)

    if(rgb):
        target_Y = 16 + (65.738 * target_data[:,:,0] + 129.057 * target_data[:,:,1] + 25.064 * target_data[:,:,2]) / 256    
        ref_Y = 16 + (65.738 * ref_data[:,:,0] + 129.057 * ref_data[:,:,1] + 25.064 * ref_data[:,:,2]) / 256  
    else:
        target_Y = target_data
        ref_Y = ref_data

    ssim = structural_similarity(im1=target_Y/255, im2=ref_Y/255, full=False)

    return ssim

def YUVFromRGB(image, withMargin=False):
    img = np.array(image, dtype=np.float32)
    yuv = np.zeros_like(img)

    # set image range 0~1
    if(img.max() > 1):
        img = img/255.

    yuv[:,:,0] = np.sum((0.299 * img[:,:,0], 0.587 * img[:,:,1], 0.114 * img[:,:,2]), axis=0)
    yuv[:,:,1] = np.sum((-0.1687 * img[:,:,0], -0.3313 * img[:,:,1], 0.5 * img[:,:,2]), axis=0)
    yuv[:,:,2] = np.sum((0.5 * img[:,:,0], -0.4187 * img[:,:,1], -0.0813 * img[:,:,2]), axis=0)

    if(withMargin):
        # yuv[:,:,0] = yuv[:,:,0] / 256
        yuv[:,:,0] = 219 * yuv[:,:,0] + 16
        # yuv[:,:,1] = yuv[:,:,1] / 256
        yuv[:,:,1] = 224 * yuv[:,:,1] + 128
        # yuv[:,:,2] = yuv[:,:,2] / 256
        yuv[:,:,2] = 224 * yuv[:,:,2] + 128

    return yuv
                    
def LSM(img_Y, img_ddsd_Y, filter=H_a):
    temp = np.zeros(filter.shape[0])
    for i in range(filter.shape[0]):
        temp[i] = np.sum(np.abs(convolve2d(img_ddsd_Y, filter[i],mode='same', boundary='fill', fillvalue=0)))
    HFE_ddsd = np.sum(temp)

    temp = np.zeros(filter.shape[0])
    for i in range(filter.shape[0]):
        temp[i] = np.sum(np.abs(convolve2d(img_Y, filter[i],mode='same', boundary='fill', fillvalue=0)))
    HFE_img = np.sum(temp)

    return HFE_img / HFE_ddsd

def LSM2(img_Y, filter=H_a):
    img = img_Y / 256

    temp = np.zeros(filter.shape[0])
    for i in range(filter.shape[0]):
        temp[i] = np.sum(np.square(convolve2d(img, filter[i],mode='same', boundary='fill', fillvalue=0)))
    HFE_img = np.sum(temp)

    mn = img_Y.shape[0]*img_Y.shape[1]

    return HFE_img / mn
    
def MyInterp(img_s, filter=H_BI):
    # interpolate by 3
    model = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=3, groups=1, bias=False, device=device)
    # filter should 5x5
    model.weight = torch.nn.Parameter(filter.expand(1,1,5,5), requires_grad=False)

    input = torch.zeros((1, img_s.shape[2], img_s.shape[0], img_s.shape[1]))
    output = np.zeros((img_s.shape[0] * 3, img_s.shape[1] * 3, img_s.shape[2]))
    
    for i in range(img_s.shape[2]):
        input[0,i,:,:] = img_s[:,:,i]
        output[:,:,i] = model(input[:,i,:,:].reshape(1,1,img_s.shape[0],img_s.shape[1]))[:,:,1:-1,1:-1]

    return output.astype(int)
     
def MyInterpBy2(img_s, filter=H_BI_2):
    # interpolate by 2
    model = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=2, groups=1, bias=False)
    # filter should 3x3
    model.weight = torch.nn.Parameter(filter.expand(1,1,3,3), requires_grad=False)

    input = torch.zeros((1, img_s.shape[2], img_s.shape[0], img_s.shape[1]))
    output = np.zeros((img_s.shape[0] * 2, img_s.shape[1] * 2, img_s.shape[2]))
    
    for i in range(img_s.shape[2]):
        input[0,i,:,:] = torch.tensor(img_s[:,:,i])
        output[:,:,i] = model(input[:,i,:,:].reshape(1,1,img_s.shape[0],img_s.shape[1]))[:,:,1:,1:]

    return output.astype(int)

def DPD_n(img, rate=3):
    height, width, channel = img.shape
    
    # DPD mask
    R_dpd = torch.zeros(height, width, dtype=int, device=device)

    R_dpd[1::rate,1::rate] = 1
    G_dpd = R_dpd.clone().detach()
    B_dpd = R_dpd.clone().detach()

    tensor_img = torch.tensor(img, device=device)
    img_dpd = torch.zeros(height, width, channel, dtype=float,device=device)

    # apply mask
    img_dpd[:, :, 0] = tensor_img[:, :, 0] * R_dpd
    img_dpd[:, :, 1] = tensor_img[:, :, 1] * G_dpd
    img_dpd[:, :, 2] = tensor_img[:, :, 2] * B_dpd
    
    return NToOne(img_dpd, rate=rate)

def DSD_n(img, rate=3):
    height, width, channel = img.shape
    
    # DPD mask
    R_dsd = torch.zeros(height, width, dtype=int, device=device)
    G_dsd = torch.zeros(height, width, dtype=int, device=device)
    B_dsd = torch.zeros(height, width, dtype=int, device=device)

    R_dsd[1::rate,0::rate] = 1
    G_dsd[1::rate,1::rate] = 1
    B_dsd[1::rate,2::rate] = 1

    tensor_img = torch.tensor(img, device=device)
    img_dsd = torch.zeros(height, width, channel, dtype=float,device=device)

    # apply mask
    img_dsd[:, :, 0] = tensor_img[:, :, 0] * R_dsd
    img_dsd[:, :, 1] = tensor_img[:, :, 1] * G_dsd
    img_dsd[:, :, 2] = tensor_img[:, :, 2] * B_dsd
    
    return NToOne(img_dsd, rate=rate)

def DDSD_n(img, rate=3):
    height, width, channel = img.shape
    
    # DDSD mask
    R_ddsd = torch.zeros(height, width, dtype=int, device=device)
    G_ddsd = torch.zeros(height, width, dtype=int, device=device)
    B_ddsd = torch.zeros(height, width, dtype=int, device=device)

    R_ddsd[0::rate,0::rate] = 1
    G_ddsd[1::rate,1::rate] = 1
    B_ddsd[2::rate,2::rate] = 1

    tensor_img = torch.tensor(img, device=device)
    img_ddsd = torch.zeros(height, width, channel, dtype=float,device=device)

    # apply mask
    img_ddsd[:, :, 0] = tensor_img[:, :, 0] * R_ddsd
    img_ddsd[:, :, 1] = tensor_img[:, :, 1] * G_ddsd
    img_ddsd[:, :, 2] = tensor_img[:, :, 2] * B_ddsd
    
    return NToOne(img_ddsd, rate=rate)

def NToOne(sampled_img, rate=3):
    height, width, channel = sampled_img.shape
    sheight = (int)(height / rate)
    swidth = (int)(width / rate)

    sampled_img = sampled_img[:sheight*rate,:swidth*rate,:]
    
    out = torch.zeros(sheight, swidth, channel, dtype=torch.float, device=device)
    
    out[:,:,0] = torch.sum(sampled_img[:,:,0].reshape(sheight,rate, swidth,rate).float(), dim=(1,3))
    out[:,:,1] = torch.sum(sampled_img[:,:,1].reshape(sheight,rate, swidth,rate).float(), dim=(1,3))
    out[:,:,2] = torch.sum(sampled_img[:,:,2].reshape(sheight,rate, swidth,rate).float(), dim=(1,3))
    
    return out

def myPDAF_n(img, rate=3):
    img = np.array(img, dtype=np.float32)
    if(img.max() > 1):
        img = (img / 255 - 1/2) * 2
    else:
        img = (img - 1/2) * 2

    # anti-aliasing filter 적용 위해 fft
    L_out, L_shift, L_fft = myfft(img, storeShift=True, storeFft=True)

    height, width, channel = img.shape
    cheight = (int)(height/2)
    cwidth = (int)(width/2)

    f_cut_vert = 1/6 * height
    f_cut_hori = 1/6 * width

    # if(showProcess):
    #     print(f_cut_vert)
    #     print(f_cut_hori)

    # low-pass mask
    mask = np.zeros(L_shift[0].shape)
    mask[cheight-(int)(f_cut_vert):cheight+(int)(f_cut_vert) + 1, cwidth-(int)(f_cut_hori):cwidth+(int)(f_cut_hori) + 1] = 1

    L_shift_pdaf = np.ndarray(shape=channel, dtype=object)
    # L_shift_pdaf_mag = np.ndarray(shape=channel, dtype=object)
    for c in range(channel):
        L_shift_pdaf[c] = L_shift[c] * mask
        # L_shift_pdaf_mag[c] = 20 * np.log(np.abs(L_shift_pdaf[c] + 1))
    
    # if(showProcess):
    #     temp = np.concatenate((L_shift_pdaf_mag[0], L_shift_pdaf_mag[1], L_shift_pdaf_mag[2]), axis=1)
    #     plt.imshow(temp, cmap='gray')

    L_pdaf = myifft(L_shift_pdaf)

    temp = np.ndarray(shape=img.shape, dtype=float)
    for c in range (3):
        temp[:,:,c] = ((L_pdaf[c]/2 + 1/2) * 255)# .astype(np.int32)
    
    L_pdaf = np.maximum(0, np.minimum(temp, 255))
    S_pdaf_dpd = DPD_n(L_pdaf, rate=rate)
    
    return S_pdaf_dpd

# Diamond Pentile
def DPD_dp(img):
    height, width, channel = img.shape
    
    img = np.array(img, dtype=np.float32)

    # DPD mask
    R_dpd = np.zeros((int(height/2), int(width/4)), dtype=int)
    G_dpd = np.zeros((int(height/2), int(width/2)), dtype=int)
    B_dpd = np.zeros((int(height/2), int(width/4)), dtype=int)

    R_dpd[0::2,:] = img[0::4,1::4,0][:R_dpd[0::2,:].shape[0],:R_dpd[0::2,:].shape[1]]
    R_dpd[1::2,:] = img[2::4,3::4,0][:R_dpd[1::2,:].shape[0],:R_dpd[1::2,:].shape[1]]

    G_dpd = img[1::2,1::2,1]

    B_dpd[0::2,:] = img[0::4,3::4,2][:B_dpd[0::2,:].shape[0],:B_dpd[0::2,:].shape[1]]
    B_dpd[1::2,:] = img[2::4,1::4,2][:B_dpd[1::2,:].shape[0],:B_dpd[1::2,:].shape[1]]

    return R_dpd, G_dpd, B_dpd

def DSD_dp(img):
    height, width, channel = img.shape
    
    img = np.array(img, dtype=np.float32)

    # room for DSD dp 
    R_dsd = np.zeros((int(height/2), int(width/4)), dtype=int)
    G_dsd = np.zeros((int(height/2), int(width/2)), dtype=int)
    B_dsd = np.zeros((int(height/2), int(width/4)), dtype=int)

    R_dsd[0::2,:] = img[0::4,0::4,0][:R_dsd[0::2,:].shape[0],:R_dsd[0::2,:].shape[1]]
    R_dsd[1::2,:] = img[2::4,2::4,0][:R_dsd[1::2,:].shape[0],:R_dsd[1::2,:].shape[1]]

    G_dsd = img[1::2,1::2,1]

    B_dsd[0::2,:-1] = img[0::4,4::4,2][:B_dsd[0::2,:-1].shape[0],:B_dsd[0::2,:-1].shape[1]]
    B_dsd[1::2,:] = img[2::4,2::4,2][:B_dsd[1::2,:].shape[0],:B_dsd[1::2,:].shape[1]]

    return R_dsd, G_dsd, B_dsd

def DDSD_dp(img):
    height, width, channel = img.shape
    
    img = np.array(img, dtype=np.float32)

    # room for DDSD dp 
    R_ddsd = np.zeros((int(height/2), int(width/4)), dtype=int)
    G_ddsd = np.zeros((int(height/2), int(width/2)), dtype=int)
    B_ddsd = np.zeros((int(height/2), int(width/4)), dtype=int)

    R_ddsd[2::2,:] = img[3:-3:4,0::4,0][:R_ddsd[2::2,:].shape[0],:R_ddsd[2::2,:].shape[1]]
    R_ddsd[1::2,:] = img[1::4,2::4,0][:R_ddsd[1::2,:].shape[0],:R_ddsd[1::2,:].shape[1]]

    G_ddsd = img[1::2,1::2,1]

    B_ddsd[0::2,:-1] = img[1::4,4::4,2][:B_ddsd[0::2,:-1].shape[0],:B_ddsd[0::2,:-1].shape[1]]
    B_ddsd[1::2,:] = img[3::4,2::4,2][:B_ddsd[1::2,:].shape[0],:B_ddsd[1::2,:].shape[1]]

    return R_ddsd, G_ddsd, B_ddsd

def PDAF_dp(img):
    img = np.array(img, dtype=np.float32)
    if(img.max() > 1):
        img = (img / 255 - 1/2) * 2
    else:
        img = (img - 1/2) * 2

    # anti-aliasing filter 적용 위해 fft
    L_out, L_shift, L_fft = myfft(img, storeShift=True, storeFft=True)

    height, width, channel = img.shape
    cheight = (int)(height/2)
    cwidth = (int)(width/2)

    f_cut_vert = 1/4 * height
    f_cut_hori = 1/4 * width

    # low-pass mask
    mask = np.zeros(L_shift[0].shape)
    mask[cheight-(int)(f_cut_vert):cheight+(int)(f_cut_vert) + 1, cwidth-(int)(f_cut_hori):cwidth+(int)(f_cut_hori) + 1] = 1

    L_shift_pdaf = np.ndarray(shape=channel, dtype=object)
    for c in range(channel):
        L_shift_pdaf[c] = L_shift[c] * mask

    L_pdaf = myifft(L_shift_pdaf)

    temp = np.ndarray(shape=img.shape, dtype=float)
    for c in range (3):
        temp[:,:,c] = ((L_pdaf[c]/2 + 1/2) * 255)
    
    L_pdaf = np.maximum(0, np.minimum(temp, 255))
    S_pdaf_dpd = DPD_dp(L_pdaf)
    
    return S_pdaf_dpd

def ConvolveDownsample(img, filter_rb, filter_g, rate=2):
    out = np.zeros_like(img[::rate,::rate,:])
    
    out[:,:,0] = convolve(img[:,:,0], filter_rb, mode='constant')[::rate,::rate]
    out[:,:,1] = convolve(img[:,:,1], filter_g, mode='constant')[::rate,::rate]
    out[:,:,2] = convolve(img[:,:,2], filter_rb, mode='constant')[::rate,::rate]

    return out

def Vimg_Stripe(img_r, img_g, img_b, IF_g=filter_g):
    if not (torch.is_tensor(img_r) or \
            torch.is_tensor(img_g) or \
            torch.is_tensor(img_b)):
        t = transforms.ToTensor()
        img_r = t(img_r).unsqueeze(0).to(device)
        img_g = t(img_g).unsqueeze(0).to(device)
        img_b = t(img_b).unsqueeze(0).to(device)

    temp_r = torch.repeat_interleave(img_r, 2, dim=-2)
    temp_r = torch.repeat_interleave(temp_r, 2, dim=-1)
    
    temp_g = torch.repeat_interleave(img_g, 2, dim=-2)
    temp_g = torch.repeat_interleave(temp_g, 2, dim=-1)
    
    temp_b = torch.repeat_interleave(img_b, 2, dim=-2)
    temp_b = torch.repeat_interleave(temp_b, 2, dim=-1)

    mask_g = torch.zeros_like(temp_g)
    mask_g[:,:,::2,::2] = 1.
    
    out_r = functional.conv2d(temp_r * mask_g, weight=IF_g, padding=1)
    out_g = functional.conv2d(temp_g * mask_g, weight=IF_g, padding=1)
    out_b = functional.conv2d(temp_b * mask_g, weight=IF_g, padding=1)

    return torch.cat((out_r, out_g, out_b), dim=1)

def Vimg_Pentile(img_r, img_g, img_b, IF_rb=filter_b, IF_g=filter_g):
    if not (torch.is_tensor(img_r) or \
            torch.is_tensor(img_g) or \
            torch.is_tensor(img_b)):
        t = transforms.ToTensor()
        img_r = t(img_r).unsqueeze(0).to(device)
        img_g = t(img_g).unsqueeze(0).to(device)
        img_b = t(img_b).unsqueeze(0).to(device)

    temp_r = torch.repeat_interleave(img_r, 2, dim=-2)
    temp_r = torch.repeat_interleave(temp_r, 4, dim=-1)
    
    temp_g = torch.repeat_interleave(img_g, 2, dim=-2)
    temp_g = torch.repeat_interleave(temp_g, 2, dim=-1)
    
    temp_b = torch.repeat_interleave(img_b, 2, dim=-2)
    temp_b = torch.repeat_interleave(temp_b, 4, dim=-1)

    try:
        temp_r1 = torch.zeros_like(temp_g)
        temp_r1[:,:,:temp_r.shape[2],:temp_r.shape[3]] = temp_r
        temp_b1 = torch.zeros_like(temp_g)
        temp_b1[:,:,:temp_b.shape[2],:temp_b.shape[3]] = temp_b
    except:
        temp_r1 = temp_r[:,:,:temp_g.shape[2],:temp_g.shape[3]]
        temp_b1 = temp_b[:,:,:temp_g.shape[2],:temp_g.shape[3]]

    mask_r = torch.zeros_like(temp_r1)
    mask_r[:,:,1::4, 1::4] = 1.
    mask_r[:,:,3::4, 3::4] = 1.

    mask_g = torch.zeros_like(temp_g)
    mask_g[:,:,::2,::2] = 1.

    mask_b = torch.zeros_like(temp_b1)
    mask_b[:,:,1::4, 3::4] = 1.
    mask_b[:,:,3::4, 1::4] = 1.
    
    out_r = functional.conv2d(temp_r1 * mask_r, weight=IF_rb, padding=3)
    out_g = functional.conv2d(temp_g * mask_g, weight=IF_g, padding=1)
    out_b = functional.conv2d(temp_b1 * mask_b, weight=IF_rb, padding=3)

    return torch.cat((out_r, out_g, out_b), dim=1)
    

def Vimg_PenTile_np(img_r,img_g,img_b, IF_rb=IF_b.cpu().numpy(), IF_g=IF_g.cpu().numpy()):
    # interpolate by 2
    img_r = img_r.astype(np.float32)
    img_g = img_g.astype(np.float32)
    img_b = img_b.astype(np.float32)
    
    temp_r = np.repeat(img_r,2,axis=0)
    temp_r = np.repeat(temp_r,4,axis=1)
        
    temp_g = np.repeat(img_g,2,axis=0)
    temp_g = np.repeat(temp_g,2,axis=1)
    
    temp_b = np.repeat(img_b,2,axis=0)
    temp_b = np.repeat(temp_b,4,axis=1)
    
    try:
        temp_r1 = np.zeros_like(temp_g)
        temp_r1[:temp_r.shape[0],:temp_r.shape[1]] = temp_r
        temp_b1 = np.zeros_like(temp_g)
        temp_b1[:temp_b.shape[0],:temp_b.shape[1]] = temp_b
    except:
        temp_r1 = temp_r[:temp_g.shape[0],:temp_g.shape[1]]
        temp_b1 = temp_b[:temp_g.shape[0],:temp_g.shape[1]]

    
    mask_r = np.zeros_like(temp_r1)
    mask_r[1::4,1::4]=1.
    mask_r[3::4,3::4]=1.
    
    mask_g = np.zeros_like(temp_g)
    mask_g[::2,::2]=1.
    
    mask_b = np.zeros_like(temp_b1)
    mask_b[1::4,3::4]=1.
    mask_b[3::4,1::4]=1.
        
    out_r = convolve(temp_r1*mask_r, IF_rb, mode='constant')
    out_g = convolve(temp_g*mask_g, IF_g, mode='constant')
    out_b = convolve(temp_b1*mask_b, IF_rb, mode='constant')
    
    return np.stack((out_r,out_g,out_b),axis=-1)

def ConvolveDownsample_PenTile_np(img, filter_rb, filter_g, rate=2):
    out_g = np.zeros_like(img[::rate,::rate,0]).astype(np.float32)
    out_r = np.zeros_like(img[::rate,::rate*2,0]).astype(np.float32)
    out_b = np.zeros_like(img[::rate,::rate*2,0]).astype(np.float32)
    
    img = img.astype(np.float32)        # 이미지 처리할 때에는 float로 변환 후 처리
    out_g = convolve(img[:,:,1], filter_g, mode='constant')[::rate,::rate]
                                                            
    temp_r = convolve(img[:,:,0], filter_rb, mode='constant')
    temp_b = convolve(img[:,:,2], filter_rb, mode='constant')

    if (out_r.shape[0]%2 == 0):
        if len(temp_r[0,1::rate*2]) == len(temp_r[0,1+rate::rate*2]):   # 가로 갯수가 짝수면
            out_r[::2,:] = temp_r[1::rate*2,1::rate*2]              # odd line
            out_r[1::2,:] = temp_r[1+rate::rate*2,1+rate::rate*2]   # even line 
            out_b[::2,:] = temp_b[1::rate*2,1+rate::rate*2]              # odd line
            out_b[1::2,:] = temp_b[1+rate::rate*2,1::rate*2]   # even line
        else:
            out_r[::2,:] = temp_r[1::rate*2,1::rate*2]              # odd line
            out_r[1::2,:-1] = temp_r[1+rate::rate*2,1+rate::rate*2]   # even line 
            out_b[::2,:-1] = temp_b[1::rate*2,1+rate::rate*2]              # odd line
            out_b[1::2,:] = temp_b[1+rate::rate*2,1::rate*2]   # even line
    else:
        if len(temp_r[0,1::rate*2]) == len(temp_r[0,1+rate::rate*2]):   # 가로 갯수가 짝수면
            out_r[:-1:2,:] = temp_r[1::rate*2,1::rate*2]              # odd line
            out_r[1::2,:] = temp_r[1+rate::rate*2,1+rate::rate*2]   # even line
            out_b[:-1:2,:] = temp_b[1::rate*2,1+rate::rate*2]              # odd line
            out_b[1::2,:] = temp_b[1+rate::rate*2,1::rate*2]   # even line
        else:
            out_r[:-1:2,:] = temp_r[1::rate*2,1::rate*2]              # odd line
            out_r[1::2,:-1] = temp_r[1+rate::rate*2,1+rate::rate*2]   # even line 
            out_b[:-1:2,:-1] = temp_b[1::rate*2,1+rate::rate*2]              # odd line
            out_b[1::2,:] = temp_b[1+rate::rate*2,1::rate*2]   # even line

    return np.clip(out_r,0,255), np.clip(out_g,0,255), np.clip(out_b,0,255)     # uint로 출력할 것이므로 overflow/underflow 없게 clip

from skimage.io import imread
from skimage.util import img_as_float
from sklearn.feature_extraction.image import extract_patches_2d
from torchvision import transforms as T
import random

class MyRotationTransform(torch.nn.Module):
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return T.functional.rotate(x, angle)

def dataGen(path, crop_size=128, batch_size=32, device='cpu', requires_grad=True, bound=False, boundRange=2, bInv=False):
    img = imread(path)
    img_tensor = T.ToTensor()(img).to(device=device)
    
    if(bound):
        if(bInv):
            bound = torch.zeros_like(img_tensor[0,:,:].unsqueeze(0))
            bound[:,boundRange-1:-(boundRange-1), boundRange-1:-(boundRange-1)] = 1
        else:
            bound = torch.ones_like(img_tensor[0,:,:].unsqueeze(0))
            bound[:,boundRange-1:-(boundRange-1), boundRange-1:-(boundRange-1)] = 0
        img_tensor = torch.cat((img_tensor, bound), dim=0)
    
    transforms = torch.nn.Sequential(
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        MyRotationTransform([0, 90,180,270]),
        T.RandomCrop((crop_size, crop_size))
    )

    patches = []
    for i in range(batch_size):
        patches.append(transforms(img_tensor))
       
    return torch.stack(patches, axis=0)

def PentileEmulation(img_r, img_g, img_b, Clip=True, Uint8=True):
    
    # 
    if(img_g.shape[1]//2 < img_r.shape[1]):
        img_g = np.pad(img_g,((0,0),(0,1)),mode='symmetric')
    if(img_g.shape[1] > img_r.shape[1]*2):
        img_g = img_g[:,:-1]

    # enlarging image
    img_enlarge_g = np.repeat(img_g, 6, axis=0)
    img_enlarge_g = np.repeat(img_enlarge_g, 6, axis=1).astype(np.float32)
    
    img_enlarge_r = np.repeat(img_r, 6, axis=0)
    img_enlarge_r = np.repeat(img_enlarge_r, 12, axis=1)
    img_enlarge_r = np.pad(img_enlarge_r,((2,0), (2,0))).astype(np.float32)
    
    img_enlarge_b = np.repeat(img_b, 6, axis=0)
    img_enlarge_b = np.repeat(img_enlarge_b, 12, axis=1)
    img_enlarge_b = np.pad(img_enlarge_b,((2,0), (2,0))).astype(np.float32)

    # mask generation
    height, width = img_enlarge_r.shape
    R_mask = np.zeros((height,width), dtype=np.float32)
    B_mask = np.zeros((height,width), dtype=np.float32)
    height, width = img_enlarge_g.shape
    G_mask = np.zeros((height,width), dtype=np.float32)

    for i in range(2,7):
        for j in range(2,7):
            if not ((i == 2 and j == 2)
            or (i == 2 and j == 6)
            or (i == 6 and j == 2)
            or (i == 6 and j == 6)
            or (i == 2 and j == 5)
            or (i == 5 and j == 6)
            or (i == 6 and j == 3)):
                R_mask[i::12,j::12] = 1
                R_mask[i+6::12,j+6::12] = 1
                B_mask[i::12,j+6::12] = 1
                B_mask[i+6::12,j::12] = 1

    for i in range(3):
        for j in range(3):
            G_mask[i::6,j::6] = 1
    
    # masking and crop
    
#     img_enlarge_r *= R_mask.astype(np.uint8)
#     img_enlarge_g *= G_mask.astype(np.uint8)
#     img_enlarge_b *= B_mask.astype(np.uint8)

    img_enlarge_r *= R_mask
    img_enlarge_g *= G_mask
    img_enlarge_b *= B_mask
    
    img_enlarge_r = img_enlarge_r[:-2,:-2]
    img_enlarge_b = img_enlarge_b[:-2,:-2]


    if(Clip):
        if(Uint8):
            return np.clip(np.stack((img_enlarge_r, img_enlarge_g, img_enlarge_b), axis=-1),0.,255.).astype(np.uint8)
        else:
            return np.clip(np.stack((img_enlarge_r, img_enlarge_g, img_enlarge_b), axis=-1),0.,255.)
    else:
        if(Uint8):
            return np.stack((img_enlarge_r, img_enlarge_g, img_enlarge_b), axis=-1).astype(np.uint8)
        else:
            return np.stack((img_enlarge_r, img_enlarge_g, img_enlarge_b), axis=-1)

def RGBToPenTile(img):
    if np.max(img) <= 1:
        img = img * 255

    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]

    # inverse gamma correction for linearlization
    img_r = (img_r/255) ** (2.2)
    img_b = (img_b/255) ** (2.2)

    # pad if width is odd
    if(img.shape[1] % 2 == 1):
        img_r = np.pad(img_r,((0,0),(0,1)),mode='symmetric')
        img_b = np.pad(img_b,((0,0),(0,1)),mode='symmetric')

    # average 2 adjacent red/blue subpixel
    img_r[:,0::2] = (img_r[:,0::2] + img_r[:,1::2])/2
    img_b[:,0::2] = (img_b[:,0::2] + img_b[:,1::2])/2
    img_rp = img_r[:,::2] 
    img_bp = img_b[:,::2] 

    # gamma correction
    img_rp = img_rp ** (1/2.2) * 255
    img_bp = img_bp ** (1/2.2) * 255

    return img_rp, img_g, img_bp


def PenTiletoRGB(pr, g, pb):
    # container same shape with g
    tr = np.zeros_like(g)
    tb = np.zeros_like(g)

    # fill red/blue subpixels, blank from below subpixels
    # if(tr.shape[0] % 2 == 0): # height is even
    if (tr.shape[1] % 2 == 0):  # width is even
        tr[::2, ::2] = pr[::2, :]
        tr[::2, 1::2] = pr[1::2, :]
        tr[1::2, ::2] = np.pad(pr[2::2, :], ((0, 1), (0, 0)), 'edge')  # pad 0 to the bottom
        tr[1::2, 1::2] = pr[1::2, :]
        tb[::2, ::2] = pb[1::2, :]
        tb[::2, 1::2] = pb[::2, :]
        tb[1::2, ::2] = pb[1::2, :]
        tb[1::2, 1::2] = np.pad(pb[2::2, :], ((0, 1), (0, 0)), 'edge')  # pad 0 to the bottom
    else:  # width is odd
        tr[::2, ::2] = pr[::2, :]
        tr[::2, 1::2] = pr[1::2, :-1]
        tr[1::2, ::2] = np.pad(pr[2::2, :], ((0, 1), (0, 0)), 'edge')  # pad 0 to the bottom
        tr[1::2, 1::2] = pr[1::2, :-1]
        tb[::2, ::2] = pb[1::2, :]
        tb[::2, 1::2] = pb[::2, :-1]
        tb[1::2, ::2] = pb[1::2, :]
        tb[1::2, 1::2] = np.pad(pb[2::2, :-1], ((0, 1), (0, 0)), 'edge')  # pad 0 to the bottom
    # else: # height is odd
    #     if(tr.shape[1] % 2 == 0):
    #         tr[::2,::2] = pr[::2,:]
    #         tr[::2,1::2] = np.pad(pr[1::2,:], ((0,1),(0,0)), 'constant') # pad 0 to the bottom
    #         tr[1::2,::2] = pr[2::2,:]
    #         tr[1::2,1::2] = pr[1::2,:]
    #         tb[::2,::2] = np.pad(pb[1::2,:], ((0,1),(0,0)), 'constant') # pad 0 to the bottom
    #         tb[::2,1::2] = pb[::2,:]
    #         tb[1::2,::2] = pb[1::2,:]
    #         tb[1::2,1::2] = pb[2::2,:]
    #     else:
    #         tr[::2,::2] = pr[::2,:]
    #         tr[::2,1::2] = np.pad(pr[1::2,:-1], ((0,1),(0,0)), 'constant') # pad 0 to the bottom
    #         tr[1::2,::2] = pr[2::2,:]
    #         tr[1::2,1::2] = pr[1::2,:-1]
    #         tb[::2,::2] = np.pad(pb[1::2,:], ((0,1),(0,0)), 'constant') # pad 0 to the bottom
    #         tb[::2,1::2] = pb[::2,:-1]
    #         tb[1::2,::2] = pb[1::2,:]
    #         tb[1::2,1::2] = pb[2::2,:-1]

    return np.stack((tr, g, tb), axis=2)


def RGBToPenTileEmulation(img):
    return PentileEmulation(RGBToPenTile(img))   

def LCM_block(img_block):
    # mean = np.mean(np.array(img_block))
    mean = np.mean(img_block)
    return np.mean(np.square(img_block-mean))

def LCM_whole(img, blocksize=30, keepBlockdata=False):
    img = np.array(img, dtype=np.float32)

    # set image range 0~1
    if(img.max() > 1):
        img = img/255.

    # calculate Y component of image
    y = np.sum((0.299 * img[:,:,0], 0.587 * img[:,:,1], 0.114 * img[:,:,2]), axis=0)

    # split image into blocks
    blocks = []

    for i in range(y.shape[0]//blocksize +1):
        for j in range(y.shape[1]//blocksize +1):
            if(blocksize * (i+1) < y.shape[0]):
                if(blocksize * (j+1) < y.shape[1]):
                    block = y[blocksize * i: blocksize * (i+1), blocksize * j: blocksize * (j+1)]
                else:
                    block = y[blocksize * i: blocksize * (i+1), blocksize * j:]
            else:
                if(blocksize * (j+1) < y.shape[1]):
                    block = y[blocksize * i: , blocksize * j: blocksize * (j+1)]
                else:
                    block = y[blocksize * i: , blocksize * j:]
            blocks.append(block)

    blocks = np.array(blocks, dtype=object)

    # calculate LCM for each block
    LCMs = []

    for block in blocks:
        LCMs.append(LCM_block(block))

    if(keepBlockdata):
        return np.mean(LCMs), LCMs
    else:
        return np.mean(LCMs)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2) # center로부터의 거리
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), data.ravel()) # 거리가 같은 점들의 합
    nr = np.bincount(r.ravel()) # 거리가 같은 점들 개수
    radialprofile = tbin / nr
    return radialprofile

def FA_DSD(image, rate=3, doFFT=True, ShowImg=False, ShowResult=False):
    img = np.array(image, dtype=np.complex64)
    h, w, c = img.shape
    if(ShowImg):
        print(img.shape)
        plt.imshow(image)
        plt.show()

    # channelwise FFT
    if(doFFT):
        ffts = np.zeros_like(img, dtype=np.complex64)
        for i in range(c):
            ffts[:,:,i] = fft2(img[:,:,i])
    else:
        ffts = img
    
    # construct spectra
    y_hat = np.sum(ffts, axis=-1)/3
    c1_hat = np.sum((a1*ffts[:,:,0], a2*ffts[:,:,1], ffts[:,:,2]), axis=0)
    c2_hat = np.sum((a2*ffts[:,:,0], a1*ffts[:,:,1], ffts[:,:,2]), axis=0)
    y_hat_shift = fftshift(y_hat)
    c1_hat_shift = fftshift(c1_hat)
    c2_hat_shift = fftshift(c2_hat)
    if(ShowImg):
        plt.figure(figsize=(16,8))
        plt.subplot(131)
        plt.imshow(20*np.log10(np.abs(y_hat_shift)), cmap='gray')
        plt.subplot(132)
        plt.imshow(20*np.log10(np.abs(c1_hat_shift)), cmap='gray')
        plt.subplot(133)
        plt.imshow(20*np.log10(np.abs(c2_hat_shift)), cmap='gray')
        plt.show()

    # average magnitude on same distance from center
    # csm: circular symmetric magnitude
    y_hat_csm = radial_profile(np.abs(y_hat_shift), [h//2, w//2])
    c1_hat_csm = radial_profile(np.abs(c1_hat_shift), [h//2, w//2])
    c1_hat_csms = np.concatenate((np.flip(c1_hat_csm[:w//rate]), c1_hat_csm[:-w//rate]))   # 1/3 지점에 c1이 있는 것 반영하여 shift
    c2_hat_csm = radial_profile(np.abs(c2_hat_shift), [h//2, w//2])
    c2_hat_csms = np.concatenate((np.flip(c2_hat_csm[:w//rate]), c2_hat_csm[:-w//rate]))

    diag = min(y_hat_csm.shape[0], c1_hat_csms.shape[0], c1_hat_csms.shape[0])   # 
    y_hat_csm = y_hat_csm[:diag]
    c1_hat_csms = c1_hat_csms[:diag]
    c2_hat_csms = c2_hat_csms[:diag]
    # print(y_hat_csm.shape, c1_hat_csm.shape, c2_hat_csm.shape)

    dummy = np.zeros_like(y_hat_csm)
    dummy[w//(rate*2)] = y_hat_csm.max()   # Nyquist frequency 표시용 더미
    
    if(ShowImg):
        plt.plot(np.concatenate((np.flip(y_hat_csm),y_hat_csm)))
        plt.plot(np.concatenate((np.flip(c2_hat_csms),c1_hat_csms)))
        plt.plot(np.concatenate((np.flip(dummy),dummy)))
        # plt.plot(y_hat_csm)
        # plt.plot(c1_hat_csms)
        # plt.plot(c2_hat_csms)
        # plt.plot(dummy)
        plt.yscale('log')
        plt.show()

    # laplace distribution

    f1 = np.nonzero(np.where(y_hat_csm<=c1_hat_csms, True, False))[0][0]
    f2 = np.nonzero(np.where(y_hat_csm<=c2_hat_csms, True, False))[0][0]
    f = min(f1,f2)
    if(ShowResult):
        print(f'Nyquist: {w//(rate*2)}, \t f_c1_h: {f1}, \t f_c2_h: {f2}, \t f_h: {f}, \t {w//(rate*2) < f}')

    return max(f, w//(rate*2))

def FA_DDSD(image, rate=3, doFFT=True, ShowImg=False, ShowResult=False):
    f = FA_DSD(image, rate, doFFT, ShowImg, ShowResult)
    w = image.shape[1]

    # consider vertically
    cond1 = w/rate*np.sqrt(2)/(np.sqrt(2)+1)  # nyquist 보다 크고, filtering 된 영역이 cutoff freqnecy 이하의 정보에 overlap 되지 않는 조건
    cond2 = 2/np.sqrt(np.pi)*cond1          # circular filter의 passband area가 rectangular filter의 area보다 커지는 조건
    cond3 = w/rate*np.sqrt(2)/2             # 대각선 overlap의 중간점

    # print(cond1, cond2, cond3, f)

    if(f < cond1):
        # print('case 1')
        return f, 'rect'
    elif(f < cond2):
        # print('case 2')
        return np.round(cond1).astype(int), 'rect'
    elif(f < cond3):
        # print('case 3')
        return f, 'circ'
    else:
        # print('case 4')
        return np.round(cond3).astype(int), 'circ'

def create_circular_mask(height, width, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (width//2, height//2)
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], width-center[0], height-center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def DSD_DDSD_FA(image, rate=3, showMask=False):
    img = np.array(image, dtype=np.float32)
    h, w, c = img.shape

    # channelwise FFT
    ffts = np.zeros_like(img, dtype=np.complex64)
    ffts_shift = np.zeros_like(img, dtype=np.complex64)
    for i in range(c):
        ffts[:,:,i] = fft2(img[:,:,i])
        ffts_shift[:,:,i] = fftshift(ffts[:,:,i])
    f_c_dsd = FA_DSD(ffts, rate=rate, doFFT=False)
    f_c_ddsd, mode_ddsd = FA_DDSD(ffts, rate=rate, doFFT=False)

    mask_dsd = np.zeros((h, w))
    mask_ddsd = np.zeros((h, w))

    mask_dsd[h//2-h//(rate*2):h//2+h//(rate*2)+1, w//2-f_c_dsd:w//2+f_c_dsd+1] = 1

    if(mode_ddsd == 'circ'):
        mask_ddsd = create_circular_mask(h, w, radius=f_c_ddsd)
    else: # if mode_ddsd == 'rect'
        mask_ddsd[max(0, h//2-f_c_ddsd):h//2+f_c_ddsd+1, max(0, w//2-f_c_ddsd):w//2+f_c_ddsd+1] = 1

    if(showMask):
        print(f'S_DSD: {h//(rate*2)*(2*f_c_dsd)}, \t S_DDSD_rect: {(2*f_c_ddsd)**2}, S_DDSD_circ: {np.pi*(f_c_ddsd**2)}')
        plt.subplot(121)
        plt.imshow(mask_dsd, cmap='gray')
        plt.subplot(122)
        plt.imshow(mask_ddsd, cmap='gray')
        plt.show()

    img_filtered_dsd = np.zeros_like(img, dtype=np.float32)
    img_filtered_ddsd = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        img_filtered_dsd[:,:,i] = ifft2(ifftshift(ffts_shift[:,:,i]*mask_dsd)).real
        img_filtered_ddsd[:,:,i] = ifft2(ifftshift(ffts_shift[:,:,i]*mask_ddsd)).real

    img_dsdFA = DSD_n(img_filtered_dsd, rate)
    img_ddsdFA = DDSD_n(img_filtered_ddsd, rate)

    return img_dsdFA.cpu().numpy(), img_ddsdFA.cpu().numpy()

def DSD_DDSD_FA_dp(image, showMask=False):
    img = np.array(image, dtype=np.float32)
    h, w, c = img.shape

    rate = 2

    # channelwise FFT
    ffts = np.zeros_like(img, dtype=np.complex64)
    ffts_shift = np.zeros_like(img, dtype=np.complex64)
    for i in range(c):
        ffts[:,:,i] = fft2(img[:,:,i])
        ffts_shift[:,:,i] = fftshift(ffts[:,:,i])
    f_c_dsd = FA_DSD(ffts, rate=rate, doFFT=False)
    f_c_ddsd, mode_ddsd = FA_DDSD(ffts, rate=rate, doFFT=False)

    mask_dsd = np.zeros((h, w))
    mask_ddsd = np.zeros((h, w))

    mask_dsd[h//2-h//(rate*2):h//2+h//(rate*2)+1, w//2-f_c_dsd:w//2+f_c_dsd+1] = 1

    if(mode_ddsd == 'circ'):
        mask_ddsd = create_circular_mask(h, w, radius=f_c_ddsd)
    else: # if mode_ddsd == 'rect'
        mask_ddsd[max(0, h//2-f_c_ddsd):h//2+f_c_ddsd+1, max(0, w//2-f_c_ddsd):w//2+f_c_ddsd+1] = 1

    if(showMask):
        print(f'S_DSD: {h//(rate*2)*(2*f_c_dsd)}, \t S_DDSD_rect: {(2*f_c_ddsd)**2}, S_DDSD_circ: {np.pi*(f_c_ddsd**2)}')
        plt.subplot(121)
        plt.imshow(mask_dsd, cmap='gray')
        plt.subplot(122)
        plt.imshow(mask_ddsd, cmap='gray')
        plt.show()

    img_filtered_dsd = np.zeros_like(img, dtype=np.float32)
    img_filtered_ddsd = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        img_filtered_dsd[:,:,i] = ifft2(ifftshift(ffts_shift[:,:,i]*mask_dsd)).real
        img_filtered_ddsd[:,:,i] = ifft2(ifftshift(ffts_shift[:,:,i]*mask_ddsd)).real

    img_dsdFA = DSD_dp(img_filtered_dsd)
    img_ddsdFA = DDSD_dp(img_filtered_ddsd)

    return img_dsdFA, img_ddsdFA

def LCM_wc(image, blocksize=30):
    img = np.array(image, dtype=np.float32)

    # set image range 0~1
    if(img.max() > 1):
        img = img/255.
    
    y = np.sum((c_rgb2ycbcr[0,0] * img[:,:,0], c_rgb2ycbcr[0,1] * img[:,:,1], c_rgb2ycbcr[0,2] * img[:,:,2]), axis=0)

    # split image into blocks
    blocks = []
    
    for i in range(1,y.shape[0]//blocksize):
        for j in range(1,y.shape[1]//blocksize):
            block = y[blocksize * (i-1): blocksize * (i), blocksize * (j-1): blocksize * (j)]
            blocks.append(block)

    blocks = np.array(blocks, dtype=object)

    # calculate LCM for each block
    LCMs = []

    for block in blocks:
        mean = np.mean(block)
        LCMs.append(np.sqrt(np.mean(np.square(block - mean))))

    return np.mean(LCMs)

def ACM_SAM_SR(img, threshold=np.array([20, 20, 20])):
    img = np.array(img, dtype=np.float32)

    N = img.shape[0] * img.shape[1]

    # set image range 0~1
    if(img.max() > 1):
        img = img/255.

    # subpixel recombination
    img_sr1 = np.zeros_like(img)
    img_sr2 = np.zeros_like(img)
        # G, B from current pixel, R from the pixel on the right
    img_sr1[:,:,0] = np.roll(img[:,:,0], -1, 1)
    img_sr1[:,:,1:] = img[:,:,1:].copy()
        # B from current pixel, R, G from the pixel on the right
    img_sr2[:,:,0] = np.roll(img[:,:,0], -1, 1)
    img_sr2[:,:,1] = np.roll(img[:,:,1], -1, 1)
    img_sr2[:,:,2] = img[:,:,2].copy()

    ycbcr_i = np.zeros(img.shape)
    ycbcr_sr1 = np.zeros(img.shape)
    ycbcr_sr2 = np.zeros(img.shape)

    # calcualte Y, Cb, Cr component of image
    for i in range(3):
        ycbcr_i[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img[:,:,0], c_rgb2ycbcr[i,1] * img[:,:,1], c_rgb2ycbcr[i,2] * img[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]
        ycbcr_sr1[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img_sr1[:,:,0], c_rgb2ycbcr[i,1] * img_sr1[:,:,1], c_rgb2ycbcr[i,2] * img_sr1[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]
        ycbcr_sr2[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img_sr2[:,:,0], c_rgb2ycbcr[i,1] * img_sr2[:,:,1], c_rgb2ycbcr[i,2] * img_sr2[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]

    # calculate absolute difference between input & sr images, channelwise
    E_sr1 = np.abs((ycbcr_i - ycbcr_sr1))
    E_sr2 = np.abs((ycbcr_i - ycbcr_sr2))
    
    M_sr1 = np.zeros(img.shape[:2])
    M_sr2 = np.zeros(img.shape[:2])
    abrupt = np.zeros(3)

    for i in range(3):
        # mark pixels with absolute difference > threshold
        M_sr1 = np.where(E_sr1[:,:,i]>threshold[i], 1, 0)
        M_sr2 = np.where(E_sr2[:,:,i]>threshold[i], 1, 0)
        # calculate average absolute difference of marked pixels
        sum_sr1 = np.sum(E_sr1[:,:,i]*M_sr1)/255
        sum_sr2 = np.sum(E_sr2[:,:,i]*M_sr2)/255
        abrupt[i] = (sum_sr1 + sum_sr2)/N

    acm = np.mean((abrupt[1], abrupt[2]))
    sam = abrupt[0]

    return acm, sam

def compute_aliasing_gray(image, edge):
    #
    img = np.array(image, dtype=np.float32)
    imedge = np.array(edge, dtype=np.float32)
    w, h = img.shape
    metric = 0
    p = img * imedge

    # construct 8 neighbors of img
    N = 8
    img_pad = np.pad(img, 1)
    img_neighbor = np.zeros((w, h, N))
    edge_pad = np.pad(imedge, 1)
    edge_neighbor = np.zeros((w, h, N))

    for i in range(N):
        d = np.round([np.cos(2*np.pi/N*i), np.sin(2*np.pi/N*i)]).astype(np.int32)   # displacement
        tmp = img_pad[1+d[0]:, 1+d[1]:]
        img_neighbor[:,:,i] = tmp[:w, :h]
        tmp = edge_pad[1+d[0]:, 1+d[1]:]
        edge_neighbor[:,:,i] = tmp[:w, :h]

        metric += np.sum(np.square(p - img_neighbor[:,:,i]*edge_neighbor[:,:,i]))

    return metric

def LAM(image_spd, image_ref):
    # set input dtype float
    img = np.array(image_spd, dtype=np.float32)
    ref = np.array(image_ref, dtype=np.float32)
    # fit image size
    w1, h1, _ = img.shape
    w2, h2, _ = ref.shape
    if(w1!=w2 or h1!=h2):
        img = img[:min(w1,w2), :min(h1,h2)]
        ref = ref[:min(w1,w2), :min(h1,h2)]
    w, h, c = img.shape
    # convert to gray
    img_gray = YUVFromRGB(img,withMargin=True)[:,:,0]
    ref_gray = YUVFromRGB(ref,withMargin=True)[:,:,0]
    # compute edge map
    edge_map = canny(ref_gray, sigma=np.sqrt(2), low_threshold=0.1, high_threshold=0.2)

    # compute aliasing metric
    metric = compute_aliasing_gray(img_gray, edge_map)
    metric_ref = compute_aliasing_gray(ref_gray, edge_map)

    return (metric-metric_ref)/(255**2*w*h)

def D_HIS(imorig, imspd):
    Norig = imorig.shape[0] * imorig.shape[1]
    Nspd = imspd.shape[0] * imspd.shape[1]

    # convert into YUV
    orig_yuv = YUVFromRGB(imorig,withMargin=True)
    spd_yuv = YUVFromRGB(imspd,withMargin=True)

    lstep = 5
    astep = 5
    bstep = 5
    edge1 = np.arange(16, 236, lstep)   # 16~235
    edge2 = np.arange(16, 241, astep)   # 16~240
    edge3 = np.arange(16, 241, bstep)   # 16~240

    # histogram distribution of original image
    orig_yhist, _ = np.histogram(orig_yuv[:, :, 0].ravel(), bins=edge1)
    orig_yhist = orig_yhist / Norig
    orig_uhist, _ = np.histogram(orig_yuv[:, :, 1].ravel(), bins=edge2)
    orig_uhist = orig_uhist / Norig
    orig_vhist, _ = np.histogram(orig_yuv[:, :, 2].ravel(), bins=edge3)
    orig_vhist = orig_vhist / Norig

    # histogram of subpixel image
    spd_yhist, _ = np.histogram(spd_yuv[:, :, 0].ravel(), bins=edge1)
    spd_yhist = spd_yhist / Nspd
    spd_uhist, _ = np.histogram(spd_yuv[:, :, 1].ravel(), bins=edge2)
    spd_uhist = spd_uhist / Nspd
    spd_vhist, _ = np.histogram(spd_yuv[:, :, 2].ravel(), bins=edge3)
    spd_vhist = spd_vhist / Nspd

    # calculate the distance
    y_diff = orig_yhist - spd_yhist
    u_diff = orig_uhist - spd_uhist
    v_diff = orig_vhist - spd_vhist
    y_dist = y_diff.dot(y_diff.T)**0.5
    u_dist = u_diff.dot(u_diff.T)**0.5
    v_dist = v_diff.dot(v_diff.T)**0.5

    # y_dist = np.sqrt(np.sum(np.square(orig_yhist - spd_yhist)))
    # u_dist = np.sqrt(np.sum(np.square(orig_uhist - spd_uhist)))
    # v_dist = np.sqrt(np.sum(np.square(orig_vhist - spd_vhist)))

    dist = np.array([y_dist, u_dist, v_dist])

    return dist

def D_Freq(imorig, imspd):
    orig_freq = np.zeros(imorig.shape)
    spd_freq = np.zeros(imspd.shape)
    Norig = imorig.shape[0] * imorig.shape[1]
    Nspd = imspd.shape[0] * imspd.shape[1]

    # convert into YUV
    orig_yuv = YUVFromRGB(imorig,withMargin=True)
    spd_yuv = YUVFromRGB(imspd,withMargin=True)

    # fft channel by channel
    for c in range(3):
        orig_freq[..., c] = np.log(np.abs(np.fft.fft2(orig_yuv[..., c])**2))
        spd_freq[..., c] = np.log(np.abs(np.fft.fft2(spd_yuv[..., c])**2))

    ystep = 1
    ustep = 1
    vstep = 1
    edge1 = np.arange(0, 50 + ystep, ystep)
    edge2 = np.arange(0, 50 + ustep, ustep)
    edge3 = np.arange(0, 50 + vstep, vstep)

    # histogram of original image
    orig_yhist, _ = np.histogram(orig_freq[:, :, 0].ravel(), bins=edge1)
    orig_yhist = orig_yhist / Norig
    orig_uhist, _ = np.histogram(orig_freq[:, :, 1].ravel(), bins=edge2)
    orig_uhist = orig_uhist / Norig
    orig_vhist, _ = np.histogram(orig_freq[:, :, 2].ravel(), bins=edge3)
    orig_vhist = orig_vhist / Norig

    # histogram of subpixel image
    spd_yhist, _ = np.histogram(spd_freq[:, :, 0].ravel(), bins=edge1)
    spd_yhist = spd_yhist / Nspd
    spd_uhist, _ = np.histogram(spd_freq[:, :, 1].ravel(), bins=edge2)
    spd_uhist = spd_uhist / Nspd
    spd_vhist, _ = np.histogram(spd_freq[:, :, 2].ravel(), bins=edge3)
    spd_vhist = spd_vhist / Nspd

    # calculate the distance
    ydist = np.sqrt(np.sum(np.square(orig_yhist - spd_yhist)))
    udist = np.sqrt(np.sum(np.square(orig_uhist - spd_uhist)))
    vdist = np.sqrt(np.sum(np.square(orig_vhist - spd_vhist)))

    dist = np.array([ydist, udist, vdist])

    return dist

def PenTileToRGB_SR1(pr, g, pb):
    # container same shape with g
    tr = np.zeros_like(g)
    tb = np.zeros_like(g)
    # SR 1 - R, B from left of G
    if(tr.shape[1] % 2 == 0): # width is even
        tr[::2,::2] = np.pad(pr[1::2,:-1],((0,0),(1,0)), 'edge') # pad to the left
        tr[::2,1::2] = pr[0::2,:]
        tr[1::2,::2] = np.pad(pr[1::2,:-1], ((0,0),(1,0)), 'edge') # pad to the left
        tr[1::2,1::2] = np.pad(pr[2::2,:], ((0,1),(0,0)), 'edge') # pad to the bottom
        tb[::2,::2] = np.pad(pb[::2,:-1], ((0,0),(1,0)), 'edge') # pad to the left
        tb[::2,1::2] = pb[1::2,:]
        tb[1::2,::2] = np.pad(pb[2::2,:-1], ((0,1),(1,0)), 'edge') # pad to the left and bottom
        tb[1::2,1::2] = pb[1::2,:]
    else: # width is odd
        tr[::2,::2] = np.pad(pr[1::2,:-1],((0,0),(1,0)), 'edge') # pad to the left
        tr[::2,1::2] = pr[0::2,:-1]
        tr[1::2,::2] = np.pad(pr[1::2,:-1], ((0,0),(1,0)), 'edge') # pad to the left
        tr[1::2,1::2] = np.pad(pr[2::2,:-1], ((0,1),(0,0)), 'edge') # pad to the bottom
        tb[::2,::2] = np.pad(pb[::2,:-1], ((0,0),(1,0)), 'edge') # pad to the left
        tb[::2,1::2] = pb[1::2,:-1]
        tb[1::2,::2] = np.pad(pb[2::2,:-1], ((0,1),(1,0)), 'edge') # pad to the left and bottom
        tb[1::2,1::2] = pb[1::2,:-1]

    return np.stack((tr, g, tb), axis=2)

def PenTileToRGB_SR2(pr, g, pb):
    # container same shape with g
    tr = np.zeros_like(g)
    tb = np.zeros_like(g)
    # SR 2 - R, B from above of G
    if(tr.shape[1] % 2 == 0): # width is even
        tr[::2,::2] = pr[::2,:]
        tr[::2,1::2] = pr[::2,:]
        tr[1::2,::2] = np.pad(pr[1::2,:-1], ((0,0),(1,0)), 'edge')
        tr[1::2,1::2] = pr[1::2,:]
        tb[::2,::2] = np.pad(pb[::2,:-1], ((0,0),(1,0)), 'edge')
        tb[::2,1::2] = pb[::2,:]
        tb[1::2,::2] = pb[1::2,:]
        tb[1::2,1::2] = pb[1::2,:]
    else: # width is odd
        tr[::2,::2] = pr[::2,:]
        tr[::2,1::2] = pr[::2,:-1]
        tr[1::2,::2] = np.pad(pr[1::2,:-1], ((0,0),(1,0)), 'edge')
        tr[1::2,1::2] = pr[1::2,:-1]
        tb[::2,::2] = np.pad(pb[::2,:-1], ((0,0),(1,0)), 'edge')
        tb[::2,1::2] = pb[::2,:-1]
        tb[1::2,::2] = pb[1::2,:]
        tb[1::2,1::2] = pb[1::2,:-1]

    return np.stack((tr, g, tb), axis=2)

def PenTileToRGB_SR3(pr, g, pb):
    # container same shape with g
    tr = np.zeros_like(g)
    tb = np.zeros_like(g)
    # SR 3 - R, B from below of G
    if(tr.shape[1] % 2 == 0): # width is even
        tr[::2,::2] = np.pad(pr[1::2,:-1], ((0,0),(1,0)), 'edge')
        tr[::2,1::2] = pr[1::2,:]
        tr[1::2,::2] = np.pad(pr[2::2,:], ((0,1),(0,0)), 'edge')
        tr[1::2,1::2] = np.pad(pr[2::2,:], ((0,1),(0,0)), 'edge')
        tb[::2,::2] = pb[1::2,:]
        tb[::2,1::2] = pb[1::2,:]
        tb[1::2,::2] = np.pad(pb[2::2,:-1], ((0,1),(1,0)), 'edge')
        tb[1::2,1::2] = np.pad(pb[2::2,:], ((0,1),(0,0)), 'edge')
    else: # width is odd
        tr[::2,::2] = np.pad(pr[1::2,:-1], ((0,0),(1,0)), 'edge')
        tr[::2,1::2] = pr[1::2,:-1]
        tr[1::2,::2] = np.pad(pr[2::2,:], ((0,1),(0,0)), 'edge')
        tr[1::2,1::2] = np.pad(pr[2::2,:-1], ((0,1),(0,0)), 'edge')
        tb[::2,::2] = pb[1::2,:]
        tb[::2,1::2] = pb[1::2,:-1]
        tb[1::2,::2] = np.pad(pb[2::2,:-1], ((0,1),(1,0)), 'edge')
        tb[1::2,1::2] = np.pad(pb[2::2,:-1], ((0,1),(0,0)), 'edge')

    return np.stack((tr, g, tb), axis=2)

def ACM_SAM_SR_DP(pr, g, pb, threshold=np.array([20, 20, 20])):
    pr = np.array(pr, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    pb = np.array(pb, dtype=np.float32)

    N = g.shape[0] * g.shape[1]

    # set image range 0~1
    if(pr.max() > 1):
        pr = pr/255.
    if(g.max() > 1):
        g = g/255.
    if(pb.max() > 1):
        pb = pb/255.

    # subpixel recombination
    img = PenTiletoRGB(pr, g, pb)
    img_sr1 = PenTileToRGB_SR1(pr, g, pb)
    img_sr2 = PenTileToRGB_SR2(pr, g, pb)
    img_sr3 = PenTileToRGB_SR3(pr, g, pb)

    ycbcr_i = np.zeros(img.shape)
    ycbcr_sr1 = np.zeros(img.shape)
    ycbcr_sr2 = np.zeros(img.shape)
    ycbcr_sr3 = np.zeros(img.shape)

    # calcualte Y, Cb, Cr component of image
    for i in range(3):
        ycbcr_i[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img[:,:,0], c_rgb2ycbcr[i,1] * img[:,:,1], c_rgb2ycbcr[i,2] * img[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]
        ycbcr_sr1[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img_sr1[:,:,0], c_rgb2ycbcr[i,1] * img_sr1[:,:,1], c_rgb2ycbcr[i,2] * img_sr1[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]
        ycbcr_sr2[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img_sr2[:,:,0], c_rgb2ycbcr[i,1] * img_sr2[:,:,1], c_rgb2ycbcr[i,2] * img_sr2[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]
        ycbcr_sr3[:,:,i] = np.sum((c_rgb2ycbcr[i,0] * img_sr3[:,:,0], c_rgb2ycbcr[i,1] * img_sr3[:,:,1], c_rgb2ycbcr[i,2] * img_sr3[:,:,2]), axis=0) * mult_rgb2ycbcr[i] + offset_rgb2ycbcr[i]

    # calculate absolute difference between input & sr images, channelwise
    E_sr1 = np.abs((ycbcr_i - ycbcr_sr1))
    E_sr2 = np.abs((ycbcr_i - ycbcr_sr2))
    E_sr3 = np.abs((ycbcr_i - ycbcr_sr3))
    
    M_sr1 = np.zeros(img.shape[:2])
    M_sr2 = np.zeros(img.shape[:2])
    M_sr3 = np.zeros(img.shape[:2])
    abrupt = np.zeros(3)

    for i in range(3):
        # mark pixels with absolute difference > threshold
        M_sr1 = np.where(E_sr1[:,:,i]>threshold[i], 1, 0)
        M_sr2 = np.where(E_sr2[:,:,i]>threshold[i], 1, 0)
        M_sr3 = np.where(E_sr3[:,:,i]>threshold[i], 1, 0)
        # calculate average absolute difference of marked pixels
        sum_sr1 = np.sum(E_sr1[:,:,i]*M_sr1)/255
        sum_sr2 = np.sum(E_sr2[:,:,i]*M_sr2)/255
        sum_sr3 = np.sum(E_sr3[:,:,i]*M_sr3)/255
        abrupt[i] = (sum_sr1 + sum_sr2 + sum_sr3)/N

    acm = np.mean((abrupt[1], abrupt[2]))
    sam = abrupt[0]

    return acm, sam

def SPA_six(image_spd, image_ori, image_pdaf):
    # coefficents for SPA score: [LCM, ACM, SAM, LAM, D_HIS(Cb), D_Freq(Cb)]
    SPA_coeff = [161.92566424, -29.26557438, 0.55713415, -65.60050165, 11.31441539, 1.4604384]

    # compute metrics
    contrast = LCM_wc(image_spd)
    abrupt_color, staircase = ACM_SAM_SR(image_spd)
    aliasing = LAM(image_spd, image_pdaf)
    _, cb_hist_dist, _ = D_HIS(image_ori, image_spd)
    _, cb_freq_dist, _ = D_Freq(image_ori, image_spd)

    # compute SPA score
    SPA_score = SPA_coeff[0] * contrast + SPA_coeff[1] * abrupt_color + SPA_coeff[2] * staircase + SPA_coeff[3] * aliasing + SPA_coeff[4] * cb_hist_dist + SPA_coeff[5] * cb_freq_dist

    return SPA_score

def SPA_six(features):
    # coefficents for SPA score: [LCM, ACM, SAM, LAM, D_HIS(Cb), D_Freq(Cb)]
    SPA_coeff = np.array([161.92566424, -29.26557438, 0.55713415, -65.60050165, 11.31441539, 1.4604384])

    # compute SPA score
    SPA_score = SPA_coeff.dot(features)

    return SPA_score