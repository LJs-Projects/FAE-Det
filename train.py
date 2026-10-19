# Train
from ultralytics import YOLO
model = YOLO('Mymodels/FAEDet.yaml')
results = model.train(data = 'Mydata/FLIR-Aligned.yaml',
                      batch = 32,
                      epochs = 200,
                      device = "0,1", 
                      optimizer = 'SGD',
                      project = 'runs/train/FLIR-Aligned' ,
                      imgsz = '640',
                      name = "001",
                      workers = 20,
                      deterministic = False,
                      #  amp = False,
                      #  freeze = 10,
                      #  resume = True,
                      )

