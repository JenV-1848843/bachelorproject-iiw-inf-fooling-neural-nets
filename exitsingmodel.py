from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# results = model.train(data="cifar10", epochs=10, imgsz=32) # train the model

model.eval()

activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

for k, v in model.named_parameters():
    print(k)

"""Output:
model.0.conv.conv.weight
model.0.conv.bn.weight
model.0.conv.bn.bias
model.1.conv.weight
model.1.bn.weight
model.1.bn.bias
model.2.cv1.conv.weight
model.2.cv1.bn.weight
...
"""
# print(model)


# layer = model.layer1[0]  # .conv1.register_forward_hook(get_activation('layer1_0_conv1'))
