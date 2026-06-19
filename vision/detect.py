from ultralytics import YOLO

model = YOLO("runs/detect/train-5/weights/last.pt")

image = "dataset/test/images/_0_9991_jpg.rf.9616a1ec9466314815ecef52e1dabaec.jpg"

results = model.predict(
    source=image,
    imgsz=640,
    conf=0.01
)

for r in results:
    if len(r.boxes) > 0:
        print("🚗 Car detected")
        print("Objects:", len(r.boxes))
        print("Confidence:", r.boxes.conf.tolist())
    else:
        print("No car detected")