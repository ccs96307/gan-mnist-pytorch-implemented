import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))


def test() -> None:
    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("GPU State:", device)

    # Model
    generator = torch.load("./checkpoints/generator_epoch_200.pth")
    generator.eval().to(device)

    # Generator
    noise = (torch.rand(16, 128) - 0.5) / 0.5
    noise = noise.to(device)

    fake_image = generator(noise)
    imgs_numpy = (fake_image.data.cpu().numpy()+1.0)/2.0
    show_images(imgs_numpy)
    plt.show()


if __name__ == "__main__":
    test()
