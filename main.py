import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

print(f"Dataset size: {len(train_dataset)}")
print(f"Number of batches: {len(train_loader)}")

def show_images(images, title):
    images = images.cpu().detach().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bridge
        b = self.bridge(p2)
        
        # Decoder
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        output = self.final(d2)
        output = self.sigmoid(output)
        
        return output

def print_layer_info(model):
    print("Layer Information:")
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.MaxPool2d)):
            print(f"{name}:")
            print(f"  Type: {type(layer).__name__}")
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                print(f"  In channels: {layer.in_channels}")
                print(f"  Out channels: {layer.out_channels}")
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                print(f"  Kernel size: {layer.kernel_size}")
                print(f"  Stride: {layer.stride}")
                print(f"  Padding: {layer.padding}")
            if isinstance(layer, nn.MaxPool2d):
                print(f"  Kernel size: {layer.kernel_size}")
                print(f"  Stride: {layer.stride}")
            print()

# Initialize the model
model = ImprovedUNet().to(device)
print(model)

# Print detailed layer information
print_layer_info(model)

# Test the model with a random input
test_input = torch.randn(1, 1, 28, 28).to(device)
with torch.no_grad():
    test_output = model(test_input)
print(f"Test input shape: {test_input.shape}")
print(f"Test output shape: {test_output.shape}")

# Visualize sample input and output
for batch in train_loader:
    images, _ = batch
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    show_images(images[:25], "Sample MNIST Images (Input)")
    show_images(outputs[:25], "Model Output")
    break