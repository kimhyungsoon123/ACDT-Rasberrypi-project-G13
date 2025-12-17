# AI learning Code
1. 1. Environment Setup and Data Access
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
```
2. Dataset Configuration for YOLO Training
```python
%%writefile /content/drive/MyDrive/data.yaml
train: /content/drive/MyDrive/cigarette_dataset/images/train
val: /content/drive/MyDrive/cigarette_dataset/images/val
test: /content/drive/MyDrive/cigarette_dataset/images/test

nc: 1
names: ['cigarette_butt']

!ls /content/drive/MyDrive/cigarette_dataset/images/train
!cat /content/drive/MyDrive/data.yaml
```
3. YOLOv5 Model Training and Model Saving
```python
%cd /content/yolov5

!python train.py \
    --img 640 \
    --batch 16 \
    --epochs 50 \
    --data /content/drive/MyDrive/data.yaml \
    --weights yolov5s.pt \
    --name cigarette_model

!cp /content/yolov5/runs/train/cigarette_model/weights/best.pt \
    /content/drive/MyDrive/cigarette_best.pt

from google.colab import files
files.download('/content/drive/MyDrive/cigarette_best.pt')
```
