import torch


class LinearDiscriminator(torch.nn.Module):
    """A discriminator network for distinguishing real images from generated data."""
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        negative_slope: float = 1e-2,
    ) -> None:
        """
        Args:
            input_size (int): The input size of linear layer.
            hidden_size (int): The hidden size of linear layer.
            negative_slope (float): A argument of LeakyReLU activation function.
        """
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=hidden_size),
            torch.nn.LeakyReLU(negative_slope=negative_slope),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.LeakyReLU(negative_slope=negative_slope),
            torch.nn.Linear(in_features=hidden_size, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): The input data.

        Returns:
            Tensor: The output prediction tensor.
        """
        return self.network(inputs)


class LinearGenerator(torch.nn.Module):
    """A generator network for generating images from random noise."""
    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 1024,
        output_size: int = 784,
    ) -> None:
        """
        Args:
            input_size (int): The size of the input tensor (random noise).
            hidden_size (int): The size of the hidden layers.
            output_size (int): The size of the output tensor (generated images).
        """
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=output_size),
            torch.nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): The input tensor (random noise).

        Returns:
            Tensor: The output tensor (generated images).
        """
        return self.network(inputs)
