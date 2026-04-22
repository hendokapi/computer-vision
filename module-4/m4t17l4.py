
# Real-Time Detection Lab
# This script builds on the previous lesson by applying the YOLOv3 object detector to a live webcam feed.
# It implements a continuous inference loop, processes frames in real-time, and profiles the performance
# by calculating and displaying the Frames Per Second (FPS).

import cv2
import numpy as np
import os
import requests
import time

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

def run_realtime_detection():
    """
    Main function to run the real-time object detection pipeline using a webcam.
    It loads a pre-trained model, captures video from the webcam, performs inference on each frame,
    calculates FPS, and displays the results in real-time.
    """
    # --- 1. Download necessary files ---
    # Using the same YOLOv3 model files as the previous lab.
    # For better real-time performance, a smaller model like YOLOv4-tiny could be used.
    # To do so, you would change these URLs and corresponding filenames.
    yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    yolo_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

    # Local filenames
    weights_file = "yolov3.weights"
    cfg_file = "yolov3.cfg"
    names_file = "coco.names"

    # Download all the required files
    download_file(yolo_weights_url, weights_file)
    download_file(yolo_cfg_url, cfg_file)
    download_file(coco_names_url, names_file)

    # --- 2. Load the Model and Class Names ---
    # Load class names
    with open(names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the YOLO model
    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNet(weights_file, cfg_file)
    if net.empty():
        print("Error: Failed to load YOLOv3 model.")
        return
    print("Model loaded successfully.")
    
    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # --- 3. Initialize Webcam ---
    print("Initializing webcam...")
    # cv2.VideoCapture(0) opens the default webcam.
    cap = cv2.VideoCapture(1)
    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam initialized.")

    # --- 4. The Main Loop for Real-Time Detection ---
    while True:
        # Read one frame from the webcam.
        # ret is a boolean indicating if the frame was read successfully.
        # frame is the captured image.
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Get the height and width of the frame
        (H, W) = frame.shape[:2]

        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # --- 5. Perform Inference and Profile FPS ---
        # Set the blob as input to the network
        net.setInput(blob)
        # Record the start time for profiling
        start_time = time.time()
        # Perform the forward pass
        layer_outputs = net.forward(output_layer_names)
        # Record the end time
        end_time = time.time()

        # Calculate inference time and FPS
        inference_time = end_time - start_time
        # Avoid division by zero
        fps = 1 / inference_time if inference_time > 0 else 0

        # --- 6. Post-process the Results ---
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # --- 7. Visualize the Results ---
        # Draw FPS on the frame
        fps_text = f"FPS: {round(fps, 2)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw bounding boxes for final detections
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                class_name = classes[class_ids[i]]
                confidence = confidences[i]
                color = (hash(class_name) % 255, hash(class_name[::-1]) % 255, hash(class_name*2) % 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 8. Display the Frame and Handle Exit ---
        # Show the processed frame in a window.
        # Note: cv2.imshow() may not work in all environments (e.g., some cloud notebooks).
        cv2.imshow("Real-Time Object Detection", frame)

        # Wait for 1ms and check if the 'q' key was pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # --- 9. Cleanup ---
    # Release the webcam capture object
    cap.release()
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

# --- Main execution block ---
if __name__ == "__main__":
    # Call the main function
    run_realtime_detection()
