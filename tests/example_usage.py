import torch

from refrakt_core.models.autoencoder import AutoEncoder
from refrakt_core.models.convnext import ConvNeXt
from refrakt_core.models.resnet import ResidualBlock, ResNet


def example_usage():
    # Create a ConvNeXt model
    convnext_model = ConvNeXt(in_channels=3, num_classes=10)

    # Create an AutoEncoder model
    autoencoder_model = AutoEncoder(input_dim=784, hidden_dim=16, type="vae")

    # Create a ResNet model
    resnet_model = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=3, num_classes=10)

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convnext_model.to_device(device)
    autoencoder_model.to_device(device)
    resnet_model.to_device(device)

    # Generate dummy inputs
    batch_size = 4
    img_size = 32
    x_convnext = torch.randn(batch_size, 3, img_size, img_size).to(device)
    x_autoencoder = torch.randn(batch_size, 784).to(device)
    x_resnet = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # Make predictions
    convnext_model.predict(x_convnext)
    autoencoder_model.predict(x_autoencoder)
    resnet_model.predict(x_resnet)

    # Get probabilities for classification models
    convnext_model.predict_proba(x_convnext)
    resnet_model.predict_proba(x_resnet)

    # Get model summaries
    print(convnext_model.summary())
    print(autoencoder_model.summary())
    print(resnet_model.summary())


example_usage()
