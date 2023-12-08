import cv2
import numpy as np
import datetime
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torchvision.models.detection as detection
import torchvision.ops as ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import time
import psutil
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from tqdm import tqdm
import os
import csv


def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

csv_output_folder = r"C:\Users\bitap\Documents\szakdoga\Output\CSV"
ensure_directory(csv_output_folder)

Numberofoutplut = 60

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]




# ====================== Initializations ======================

# Initialize and load YOLO
def init_yolo():
    weights_path = "./yolov3.weights"
    cfg_path = "./yolov3.cfg"
    net = cv2.dnn.readNet(weights_path, cfg_path)
    # Load coco.names classes
    coco_names_path = "./coco.names"
    classes = []
    with open(coco_names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

net_yolo, classes_yolo = init_yolo()


def init_model2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the new weights enum
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()
    transform = T.Compose([T.ToTensor()])
    return model, device, transform

model2, device2, transform2 = init_model2()

def init_ssd():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the new weights enum
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    model.eval()
    transform = T.Compose([T.ToTensor()])
    return model, device, transform

model_ssd, device_ssd, transform_ssd = init_ssd()





def init_maskrcnn():
    # Load the pre-trained Mask R-CNN model using the new weights parameter
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    return model



def init_retinanet():
    # Load the pre-trained RetinaNet model
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model


def init_mobilenet_ssd():
    # Load the pre-trained MobileNet V2 SSD model
    model = detection.ssd300_vgg16(pretrained=True)
    model.eval()
    return model




# ====================== Video Processing ======================

# Load video
video_path = "./dashcamtest.mp4"
cap = cv2.VideoCapture(video_path)

# Check video
if not cap.isOpened():
    print("Error opening video stream or file")

# Extract frames
frames = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
interval = max(1, frame_count // Numberofoutplut)  # Ensure interval is at least 1

for i in range(frame_count):
    ret, frame = cap.read()
    if i % interval == 0:
        frames.append(frame)

cap.release()


# ====================== Model Processing ======================

# For YOLO processing
def process_yolo(frame):
    # Use the same blob creation as before
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_yolo.setInput(blob)
    layer_names = net_yolo.getUnconnectedOutLayersNames()
    outs = net_yolo.forward(layer_names)
    
    class_ids = []
    confidences = []
    boxes = []
    detections = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x, y = int(center_x - width / 2), int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Use NMS to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in indices.flatten():
        box = boxes[i]
        x, y, width, height = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = classes_yolo[class_ids[i]]
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Append to detections
        detections.append((label, x, y, width, height, confidences[i]))

    return frame, detections



# Faster R-CNN processing
def process_fasterrcnn(frame):
    # Convert frame to PIL Image and apply transformations
    img = Image.fromarray(frame)
    img_t = transform2(img).unsqueeze(0).to(device2)
    
    # Get the detections from the model
    with torch.no_grad():
        prediction = model2(img_t)

    detections = []
    # Draw bounding boxes and labels on the frame
    for i in range(len(prediction[0]['labels'])):
        score = prediction[0]['scores'][i].item()
        if score > 0.5:
            box = prediction[0]['boxes'][i].tolist()
            label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]['labels'][i].item()]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Append to detections
            detections.append((label, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1]), score))

    return frame, detections




def process_ssd(frame):
    # Convert frame to PIL Image and apply transformations
    img = Image.fromarray(frame)
    img_t = transform_ssd(img).unsqueeze(0).to(device_ssd)
    
    # Get the detections from the model
    with torch.no_grad():
        prediction = model_ssd(img_t)
    
    # Apply NMS
    keep = ops.nms(prediction[0]['boxes'], prediction[0]['scores'], 0.3)
    
    detections = []
    
    # Draw bounding boxes and labels on the frame
    for i in keep:
        score = prediction[0]['scores'][i].item()
        if score > 0.2:  # Adjusted confidence threshold
            box = prediction[0]['boxes'][i].tolist()
            label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]['labels'][i].item()]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Append detection results
            detections.append((label, int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1]), score))

    return frame, detections

def process_maskrcnn(frame):
    # Convert the image to a tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(frame)

    # Get predictions from the model
    with torch.no_grad():
        prediction = model_maskrcnn([image_tensor])

    # Extract boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    detections = []

    # Convert image tensor back to numpy array for drawing
    frame_np = frame.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Threshold for detection
            box = box.int().tolist()  # Convert to integer coordinates
            cv2.rectangle(frame_np, (box[0], box[1]), (box[2], box[3]), (100, 255, 150), 2)

            label_text = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            cv2.putText(frame_np, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 2)

            # Append detection results
            detections.append((label_text, box[0], box[1], box[2] - box[0], box[3] - box[1], score.item()))

    return frame_np, detections



def process_retinanet(frame):
    # Convert the image to a tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(frame)

    # Get predictions from the model
    with torch.no_grad():
        prediction = model_retinanet([image_tensor])

    # Extract boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    detections = []

    # Convert image tensor back to numpy array for drawing
    frame_np = frame.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Threshold for detection
            box = box.int().tolist()  # Convert to integer coordinates
            cv2.rectangle(frame_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            label_text = f"Class {label.item()}"  # Adjust as per your class labels
            cv2.putText(frame_np, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Append detection results
            detections.append((label_text, box[0], box[1], box[2] - box[0], box[3] - box[1], score.item()))

    return frame_np, detections

def process_mobilenet_ssd(frame):
    # Transform the frame as required by the model
    transform = T.Compose([T.ToTensor()])
    frame_t = transform(frame).unsqueeze(0)  # Add batch dimension

    # Perform detection
    with torch.no_grad():
        prediction = model_mobilenet_ssd(frame_t)

    # Process prediction to extract bounding boxes, labels, and scores
    detections = []
    for i in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][i].tolist()
        score = prediction[0]['scores'][i].item()
        if score > 0.25:  # Confidence threshold
            label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]['labels'][i].item()]
            detections.append((label, box[0], box[1], box[2] - box[0], box[3] - box[1], score))

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detections



# ====================== Main Loop ======================

models = {
    "YOLO": process_yolo,
    "Faster R-CNN": process_fasterrcnn,
    "SSD": process_ssd,
    "Mask R-CNN": process_maskrcnn,
    "Retinanet": process_retinanet,
    "MobileNet SSD": process_mobilenet_ssd,
}

create_jpeg = True

# Dictionaries to store metrics
processing_times = {}
avg_frame_processing_times = {}
cpu_usages = {}
memory_usages = {}

model_retinanet = init_retinanet()
model_maskrcnn = init_maskrcnn()
model_mobilenet_ssd = init_mobilenet_ssd()

process = psutil.Process()
process.cpu_percent()  # Initialize the first call to ignore the first result as it can be misleading

# Main loop with modifications to save detection results in CSV files
for model_name, model_func in models.items():
    print(f"Processing with {model_name}...")
    start_time = time.time()

    cpu_usage_list = []
    memory_usage_list = []

    # Path for the CSV file for this model's predictions
    csv_output_path = os.path.join(csv_output_folder, f"{model_name}_predictions.csv")

    # Ensure the CSV output folder exists
    ensure_directory(csv_output_folder)

    with open(csv_output_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header
        csv_writer.writerow(["image_name", "label_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "confidence_score"])
       
        for idx, original_frame in tqdm(enumerate(frames), total=len(frames), desc=f"Processing {model_name}"):
            frame_start_time = time.time()
            frame = original_frame.copy()
            processed_frame, detections = model_func(frame)  # Call updated model function

            # Measure CPU and Memory usage during frame processing
            cpu_usage_list.append(process.cpu_percent())
            memory_usage_list.append(process.memory_info().rss)  # RSS gives the actual memory usage

            #generating pictures
            if create_jpeg:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                output_image_name = f"{model_name}_Frame-{idx}_{timestamp}.jpg"
                output_path = os.path.join("C:/Users/bitap/Documents/szakdogaTeszt/output", output_image_name)
                cv2.imwrite(output_path, processed_frame)


            # Use 'frame_x' format for CSV entries
            csv_image_name = f"frame_{idx}.jpg"

            # Write detections to CSV
            for detection in detections:
                label, x, y, width, height, confidence = detection
                csv_writer.writerow([csv_image_name, label, x, y, width, height, confidence])

            frame_end_time = time.time()
            avg_frame_processing_times[model_name] = (frame_end_time - frame_start_time)

    end_time = time.time()
    processing_times[model_name] = end_time - start_time

    # Compute average CPU and Memory usage for this model
    avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
    avg_memory_usage = sum(memory_usage_list) / len(memory_usage_list)

    cpu_usages[model_name] = avg_cpu_usage
    memory_usages[model_name] = avg_memory_usage

    print("Model Comparison Results\n")
    print("========================\n\n")

    print(f"Model: {model_name}\n")
    print(f"Total Processing Time: {processing_times[model_name]:.2f} seconds\n")
    print(f"Average Frame Processing Time: {avg_frame_processing_times[model_name]:.2f} seconds\n")
    print(f"CPU Usage: {cpu_usages[model_name]:.2f}%\n")
    print(f"Memory Usage: {memory_usages[model_name]/(1024**2):.2f} MB\n")  # Convert bytes to MB
    print("------------------------\n")



cv2.destroyAllWindows()



