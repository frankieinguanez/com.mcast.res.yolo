from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8n.pt")

# from PIL
im1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

for result in results:
    # Detection
    result.boxes.xyxy   # box with xyxy format, (N, 4)
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf   # confidence score, (N, 1)
    result.boxes.cls    # cls, (N, 1)

    # Segmentation
    result.masks.data      # masks, (N, H, W)
    result.masks.xy        # x,y segments (pixels), List[segment] * N
    result.masks.xyn       # x,y segments (normalized), List[segment] * N

    # Classification
    result.probs     # cls prob, (num_class, )

# Each result is composed of torch.Tensor by default,
# in which you can easily use following functionality:
result = result.cuda()