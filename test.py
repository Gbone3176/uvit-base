import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def add_noise(image, num_step):
    # 生成高斯噪声基础
    gaussian_noise = 255 * np.random.normal(0, 4, image.shape)  # 创建标准高斯噪声
    # 生成线性变化的强度
    linear_coeff = np.linspace(0, 1, num=num_step)
    # 生成从-pi到0的等间隔数组
    cosine_coeff = (np.cos(np.linspace(-1 * np.pi, 0, num_step))+ 1) / 2

    linear = []
    cosine = []
    for i in range(num_step):
        noisy_image_linear = image + (gaussian_noise * linear_coeff[i])
        noisy_image_linear = np.clip(noisy_image_linear, 0, 255).astype(np.uint8)
        noisy_image_cos = image + (gaussian_noise * cosine_coeff[i])
        noisy_image_cos = np.clip(noisy_image_cos, 0, 255).astype(np.uint8)  # 确保值在范围内
        linear.append(noisy_image_linear)
        cosine.append(noisy_image_cos)
    return linear, cosine


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_images(img_stack, title):
    # 创建一个指定大小的图形
    plt.figure(figsize=(len(img_stack) * 3, 3))

    # 使用 GridSpec 创建无间隙的子图布局
    gs = gridspec.GridSpec(1, len(img_stack), wspace=0, hspace=0)

    # 显示每幅图像
    for i in range(len(img_stack)):
        ax = plt.subplot(gs[i])  # 使用 GridSpec 选择子图位置
        ax.imshow(img_stack[i], cmap='gray')
        ax.axis('off')  # 关闭坐标轴

    # 设置标题以及保存图像
    plt.suptitle(title)  # 使用 suptitle 设置整体标题
    plt.savefig(r"E:\FDUUU\Lab\EMBC\code\U-ViT\assets\testImgs" + '\\' + f'{title}.png',
                bbox_inches='tight', dpi=300)
    plt.show()


def main():
    image_path = r'E:\FDUUU\Lab\EMBC\code\U-ViT\assets\testImgs\img1.jpg'  # 更改为您的图像路径
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lin_noise_img, cos_noise_img = add_noise(original_image, num_step=7)

    # 可视化
    visualize_images(lin_noise_img, "Linear")
    visualize_images(cos_noise_img, "Cosine")

def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    a = s.view(-1, *extra_dims)
    return s.view(-1, *extra_dims) * ts


if __name__ == '__main__':
    s = np.array([2.0, 3.0])
    ts = torch.tensor([[[1.0, 4.0], [2.0, 5.0]],[[1.0, 4.0], [2.0, 5.0]]])

    result = stp(s, ts)
    print(result)