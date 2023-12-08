# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
RUN pip install opencv-python Pillow tqdm psutil

# Install the correct version of torchvision
RUN pip install torchvision

# Copy the current directory contents into the container at /usr/src/app
COPY ["Main copy.py", "dashcamtest.mp4", "yolov3.cfg", "yolov3.weights", "coco.names", "./"]

# Run your script when the container launches
CMD ["python", "./Main copy.py"]
