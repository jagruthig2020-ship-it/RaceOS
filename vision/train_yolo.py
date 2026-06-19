from ultralytics import YOLO

# initialize model (use a pretrained YOLOv8 nano weights or change path as needed)
model = YOLO('yolov8n.pt')

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=416
)