import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import transforms

from models import LinearDiscriminator, LinearGenerator


def save_images(images: np.ndarray, epoch: int, directory="snapshots") -> None:
    """Save the images to the specified directory.
    
    Args:
        images (numpy.ndarray): An array of images to be saved.
        epoch (int): The current epoch number, used in naming the image files.
        directory (str, optional): The directory where images will be saved. Defaults to 'snapshots'.
    """

    # Check the directory is existed or not
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Prepare images
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    fig, ax = plt.subplots(sqrtn, sqrtn, figsize=(sqrtn, sqrtn))

    for index, image in enumerate(images):
        ax_index = np.unravel_index(index, (sqrtn, sqrtn))
        ax[ax_index].imshow(image.reshape(28, 28), cmap="gray")
        ax[ax_index].axis("off")

    # Padding
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)

    # Save images
    file_name = f"epoch_{epoch}.png"
    plt.savefig(os.path.join(directory, file_name))


def main() -> None:
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"GPU State: {device}")

    # Models - Generator & Discriminator
    generator = LinearGenerator().to(device)
    discriminator = LinearDiscriminator().to(device)

    # Show model architecture
    print("=========== Generator ==============")
    print(generator)
    print("\n")
    print("========= Discriminator ============")
    print(discriminator)

    # Loss function
    discriminator_criterion = torch.nn.BCELoss()
    generator_criterion = torch.nn.BCELoss()

    # Hyper-parameters
    epochs = 200
    lr = 0.0002
    batch_size = 64
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Image Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load data
    train_set = datasets.MNIST("mnist/", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Train
    for epoch in range(1, epochs + 1):
        for times, data in enumerate(train_loader, 1):
            # Real data
            real_inputs = data[0].to(device)
            real_inputs = real_inputs.view(-1, 784)

            # Real data input Discriminator
            real_outputs = discriminator(real_inputs)

            # Real data labels
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            # Fake data
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            # Generate fake images
            fake_inputs = generator(noise)

            # Fake data input Discriminator
            fake_outputs = discriminator(fake_inputs)

            # Fake data labels
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            # Concat data and labels (real & fake)
            outputs = torch.cat((real_outputs, fake_outputs), dim=0)
            targets = torch.cat((real_label, fake_label), dim=0)

            # Zero the parameter gradients
            d_optimizer.zero_grad()

            # Update Discriminator
            d_loss = discriminator_criterion(outputs, targets)
            d_loss.backward()
            d_optimizer.step()

            # Generator
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            # Fake data & labels
            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs)
            fake_labels = torch.ones(real_inputs.shape[0], 1)

            g_loss = generator_criterion(fake_outputs, fake_labels.to(device))
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if times % 100 == 0 or times == len(train_loader):
                print(f"[{epoch}/{epochs}, {times}/{len(train_loader)}] D_loss: {d_loss.item():.3f}  G_loss: {g_loss.item():.3f}")

        # Save images
        imgs_numpy = (fake_inputs.data.cpu().numpy() + 1.0) / 2.0
        save_images(
            images=imgs_numpy[:16],
            epoch=epoch,
        )

        # Save model
        if epoch % 50 == 0:
            # Check the directory is existed or not
            model_save_dir = "./checkpoints/"
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            model_save_path = os.path.join(model_save_dir, f"generator_epoch_{epoch}.pth")
            torch.save(
                obj=generator,
                f=model_save_path,
            )
            print(f"{model_save_path} saved.")

    print("Training Finished.")


if __name__ == "__main__":
    main()
