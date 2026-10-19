from ultralytics import YOLO

model = YOLO("best_flir.pt")

results = model.predict(
    source=['DATASET/FLIR/images/test/FLIR_09025_PreviewData.jpg',
            'DATASET/FLIR/image/test/FLIR_09025_PreviewData.jpeg'],
    save=True,                  
    save_txt=True,              
    save_conf=True              
)