
from ultralytics import YOLO 
model = YOLO('\Best_FLIR-Aligned.pt')
# print(model)
    
metrics = model.val(data='Mydata/FLIR-Aligned.yaml',
                    split='test',
                    device ="0",
                    imgsz=640,
                    batch=16,
                    )
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP75: {metrics.box.map75:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")