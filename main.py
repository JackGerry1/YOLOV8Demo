from ultralytics import YOLO
import os

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the image and save directories
image_directory = f'{CURRENT_DIR}/AlpacaTest-1/test/images/'
save_directory = f'{CURRENT_DIR}/AlpacaTest-1/predictions/'

# Load the pretrained YOLOv8 model (recommended)
model = YOLO('yolov8n.pt') 

# Create a new YOLO model from scratch
#model = YOLO('yolov8n.yaml') 

# Train the model on custom dataset
model.train(data=f'{CURRENT_DIR}/AlpacaTest-1/data.yaml', epochs=50, imgsz=640)

# Train Model on Coco8 dataset
#model.train(data='coco8.yaml', epochs=50, imgsz=640)

# Validate the model
model.val()

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)


# Test On Single Image
#results = model(
#    source="https://ultralytics.com/images/bus.jpg", # this image can be changed 
#    save=True, 
#    project=save_directory, 
#    name='.',        
#    exist_ok=True    
#)

### Test On Multiple Images ###

results = model(
    source=image_directory, 
    save=True, 
    project=save_directory, 
    name='.',        # Prevents creation of subfolders
    exist_ok=True    # Allows saving in an existing directory
)
