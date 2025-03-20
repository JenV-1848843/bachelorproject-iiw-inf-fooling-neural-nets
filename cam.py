import numpy as np
import cv2
from torchvision import models, transforms
from torch.nn import functional as F

def rescaleImage(image, scalePercent):
    width = int(image.shape[1] * scalePercent / 100)
    height = int(image.shape[0] * scalePercent / 100)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

model = models.resnet18(pretrained=True)
model = model.eval()

# define a function to store the feature maps at the final convolutional layer
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.layer4.register_forward_hook(getActivation('final_conv'))

# you can use any image, for this example, we'll use an image of a dog
image_path = "dog.png"

# define the data transformation pipeline: resize => tensor => normalize
transforms = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ]
)

image = cv2.imread(image_path)
orig_image = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

# apply the image transforms
image_tensor = transforms(image)
# add batch dimension
image_tensor = image_tensor.unsqueeze(0)
# forward pass through model
outputs = model(image_tensor)
# get the idx of predicted class
class_idx = F.softmax(outputs).argmax().item()
print(class_idx) # output: 201

# Fetch feature maps at the final convolutional layer
conv_features = activation['final_conv']

# Fetch the learned weights at the final feed-forward layer
weight_fc = model.fc.weight.detach().numpy()


def calculate_cam(feature_conv, weight_fc, class_idx):
    # generate the class activation maps upsample to 224x224
    size_upsample = (224, 224)

    bz, nc, h, w = feature_conv.shape

    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)

    return output_cam


# generate class activation mapping
class_activation_map = calculate_cam(conv_features, weight_fc, class_idx)


def visualize_cam(class_activation_map, width, height, orig_image):
    heatmap = cv2.applyColorMap(cv2.resize(class_activation_map, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + orig_image * 0.5

    cv2.imshow("title", rescaleImage(result, 30))
    cv2.waitKey(0)


# visualize result
visualize_cam(class_activation_map, width, height, orig_image)


