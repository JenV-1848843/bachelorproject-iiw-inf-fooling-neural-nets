from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

results = model.train(data="cifar10", epochs=10, imgsz=32) # train the model