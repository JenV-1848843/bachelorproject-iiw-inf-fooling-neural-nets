import torchvision
import os.path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchsummary import summary
from PIL import Image


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
print(device)


def showImage(image):
    # image.permute(1, 2, 0)
    denormalized_image = image / 2 + 0.5
    plt.imshow(denormalized_image)
    plt.axis('off')
    plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    )
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_set = torchvision.datasets.ImageNet(root='./data', train=True, download=True, transform=transform)
# test_set = torchvision.datasets.ImageNet(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=False)

# to_pil = transforms.ToPILImage()
# pil_img = to_pil(train_set[0][0])
# print(train_loader.dataset[0][0].permute(1, 2, 0))
# showImage(train_loader.dataset[0][0].permute(1, 2, 0))


# rgba_img = pil_img.convert("RGBA")

# plt.imshow(pil_img)
# plt.axis("off")  # Hide axis
# plt.show()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# if os.path.exists("./model.pht"):
net = torch.load("./model.pht", weights_only=False)
print(type(net))
'''
for param in net.parameters():
    param.requires_grad() == False
    '''
print("Existing model found and loaded")
'''
else:
    net = ConvNeuralNet()
    activation_maps = {}
    net.to(device)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # for param in net.parameters():
    #   print(param.grad)
    #   print()

    epochs = 10
    for epoch in range(epochs):

        running_loss = 0.0

        """
        enumerate is just used to keep track of the index of the item we're working with
        data is a list of batches of images with their corresponding labels
        every image batch has 4 dimensions:
        the first is the amount of images in the batch
        the second is the amount of channels of the image (so 3 for rgb)
        the third and fourth are the pixel dimensions of the images themselves
        """
        for i, data in enumerate(train_loader):
            # data = next(iter(train_loader))

            """
            one data batch here is a list of length 2
            the first item is a tensor of 4 image matrices
            the second item is a tensor of the 4 corresponding label ints of those images
            """
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad() # reset the gradient of optimized tensors
            # print("-----------------------------------")
            # print("now printing zero grad parameters")
            # print("-----------------------------------")

            # for param in net.parameters():
            #     print(param.grad)
            #     print()

            outputs = net(inputs) #
            # print("-----------------------------------")
            # print("now printing outputs")
            # print("-----------------------------------")
            # print (outputs)


            loss = loss_function(outputs, labels)
            # print("-----------------------------------")
            # print("now printing loss")
            # print("-----------------------------------")
            # print (loss)

            # activation_map = activation_maps['conv1'].squeeze(0)

            # print("-----------------------------------")
            # print("now printing activation map")
            # print("-----------------------------------")
            # print (activation_map)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), "./model.pht")
    '''


def view_classification(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    image = image.permute(1, 2, 0)
    denormalized_image= image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


images, _ = next(iter(test_loader))

image = images[3]
batched_image = image.unsqueeze(0).to(device)
with torch.no_grad():
    log_probabilities = net(batched_image)

probabilities = torch.exp(log_probabilities).squeeze().cpu()
view_classification(image, probabilities)
print(probabilities.data.numpy().squeeze())

'''
# Transformations for the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


# Function to show images
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # unnormalize
    plt.imshow(img)
    plt.show()


# Get some images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Display images
imshow(torchvision.utils.make_grid(images))

# Load pretrained ResNet18
model = resnet18(weights=True)
model.eval()  # Set the model to evaluation mode

# Hook setup
activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


# Register hooks
model.layer1[0].conv1.register_forward_hook(get_activation('layer1_0_conv1'))
model.layer4[0].conv1.register_forward_hook(get_activation('layer4_0_conv1'))

# Run the model
with torch.no_grad():
    output = model(images)


# Visualization function for activations
def plot_activations(layer, num_cols=4, num_activations=16):
    num_kernels = layer.shape[1]
    fig, axes = plt.subplots(nrows=(num_activations + num_cols - 1) // num_cols, ncols=num_cols, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            ax.imshow(layer[0, i].cpu().numpy(), cmap='twilight')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


# Display a subset of activations
plot_activations(activations['layer1_0_conv1'], num_cols=4, num_activations=16)
'''
