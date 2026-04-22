
# Object Detection Lab
# This script demonstrates how to use a pre-trained YOLOv3 model for object detection in an image.
# It covers loading the model, pre-processing the image, running inference, and post-processing the results,
# including Non-Maximum Suppression (NMS), to get clean detections.

import cv2
import numpy as np
import os
import requests
import matplotlib.pyplot as plt

def download_file(url, filename):
    """
    Downloads a file from a given URL if it does not already exist.
    
    Parameters:
    url (str): The URL of the file to download.
    filename (str): The local path to save the file.
    """
    # Check if the file already exists
    if not os.path.exists(filename):
        # Print a message that the file is being downloaded
        print(f"Downloading {filename}...")
        try:
            # Make the request to the URL
            response = requests.get(url, stream=True)
            # Raise an exception for bad status codes
            response.raise_for_status()
            # Open the file in binary write mode
            with open(filename, "wb") as f:
                # Write the content of the response to the file
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Print a message that the download is complete
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            # Print an error message if the download fails
            print(f"Error downloading {filename}: {e}")
            # Exit the script if a required file cannot be downloaded
            exit()

def run_object_detection():
    """
    Main function to run the object detection pipeline.
    It downloads model files and a test image, loads the model, processes the image,
    and visualizes the detection results.
    """
    # --- 1. Download necessary files ---
    # URLs for the YOLOv3 model files and a sample image
    yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    yolo_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    image_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"

    # Local filenames
    weights_file = "yolov3.weights"
    cfg_file = "yolov3.cfg"
    names_file = "coco.names"
    image_file = "dog_test_image.jpg"

    # Download all the required files
    download_file(yolo_weights_url, weights_file)
    download_file(yolo_cfg_url, cfg_file)
    download_file(coco_names_url, names_file)
    download_file(image_url, image_file)

    # --- 2. Load the Model and Class Names ---
    # Load class names from the coco.names file
    with open(names_file, "r") as f:
        # Read all lines, strip whitespace, and store in a list
        classes = [line.strip() for line in f.readlines()]

    # Load the pre-trained YOLOv3 model using OpenCV's DNN module
    # cv2.dnn.readNet expects the paths to the weights and config files
    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNet(weights_file, cfg_file)
    # Check if the network was loaded successfully
    if net.empty():
        print("Error: Failed to load YOLOv3 model.")
        return
    print("Model loaded successfully.")

    # Get the names of the output layers from the network
    # These are the layers that produce the final detection results
    output_layer_names = net.getUnconnectedOutLayersNames()

    # --- 3. Load and Prepare the Image ---
    # Load the input image from disk
    image = cv2.imread(image_file)
    # Get the height and width of the image
    (H, W) = image.shape[:2]

    # Create a "blob" from the image for the network input
    # This function resizes the image to 416x416, scales pixel values by 1/255,
    # and swaps the Blue and Red channels (as OpenCV uses BGR and YOLO was trained on RGB).
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # --- 4. Perform Inference (Forward Pass) ---
    # Set the blob as the input to the network
    net.setInput(blob)
    # Perform a forward pass through the network to get the outputs
    print("Running forward pass...")
    layer_outputs = net.forward(output_layer_names)
    print("Forward pass complete.")

    # --- 5. Post-process the Results ---
    # Initialize lists to store detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the layer outputs
    for output in layer_outputs:
        # Loop over each of the detections in the output
        for detection in output:
            # The first 5 elements are [center_x, center_y, width, height, objectness_confidence]
            # The rest are class scores for each of the 80 COCO classes
            scores = detection[5:]
            # Find the index of the class with the highest score
            class_id = np.argmax(scores)
            # Get the confidence of that class
            confidence = scores[class_id]

            # Filter out weak detections by ensuring the confidence is greater than a threshold
            if confidence > 0.5:
                # Scale the bounding box coordinates back to the original image size
                box = detection[0:4] * np.array([W, H, W, H])
                # Destructure the box coordinates
                (centerX, centerY, width, height) = box.astype("int")
                # Calculate the top-left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add the bounding box, confidence, and class ID to their respective lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # --- 6. Apply Non-Maximum Suppression (NMS) ---
    # NMS suppresses weak, overlapping bounding boxes to get clean final detections
    # It requires the boxes, confidences, a confidence threshold, and an NMS threshold
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # --- 7. Visualize the Results ---
    # Ensure at least one detection exists
    if len(indices) > 0:
        # Loop over the indices that NMS kept
        for i in indices.flatten():
            # Get the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # Get the class name and confidence
            class_name = classes[class_ids[i]]
            confidence = confidences[i]
            # Define a color for the bounding box (using a simple hash on class name for variety)
            color = (hash(class_name) % 255, hash(class_name[::-1]) % 255, hash(class_name*2) % 255)
            # Draw the bounding box rectangle on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # Create the label text with class name and confidence
            text = f"{class_name}: {confidence:.2f}"
            # Draw the label text above the bounding box
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the final image with detections
    print("Displaying detection results. Close the image window to exit.")
    # Convert image from BGR (OpenCV default) to RGB for Matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a new figure for plotting
    plt.figure(figsize=(10, 8))
    # Show the image
    plt.imshow(image_rgb)
    # Turn off the axes
    plt.axis('off')
    # Set a title for the window
    plt.title("YOLO Object Detections")
    # Show the plot
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    # Call the main function to run the detection
    run_object_detection()
