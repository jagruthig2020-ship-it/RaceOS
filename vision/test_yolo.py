from ultralytics import YOLO

model = YOLO("runs/detect/train-5/weights/last.pt")

results = model.predict(
    source="dataset/test/images/_0_9991_jpg.rf.9616a1ec9466314815ecef52e1dabaec.jpg",
    conf=0.01,
    imgsz=640,
    save=True
)

print(results[0].boxes)