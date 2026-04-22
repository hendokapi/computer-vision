
# Simple Tracker Lab
# This script implements a simple but functional multi-object tracker from scratch.
# It uses the "Tracking-by-Detection" paradigm, where an object detector (YOLOv3)
# first detects objects, and then a custom Centroid Tracker associates these detections
# across frames to maintain a unique ID for each object.

# Note: This script requires the 'scipy' library. 
# You can install it by running: pip install scipy
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import os
import requests
import time

class CentroidTracker:
    """
    A simple centroid-based object tracker.
    This class manages the registration, deregistration, and update of tracked objects.
    Data association is performed by calculating the Euclidean distance between centroids.
    """
    def __init__(self, maxDisappeared=50):
        """
        Initializes the tracker.
        
        Parameters:
        maxDisappeared (int): The maximum number of consecutive frames an object can be 'lost'
                              before it is deregistered.
        """
        self.nextObjectID = 0
        # This now stores the full bounding box for each object: {objectID: (startX, startY, endX, endY)}
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, rect):
        """
        Registers a new object with a new object ID and its bounding box.
        
        Parameters:
        rect (tuple): The (startX, startY, endX, endY) bounding box of the new object.
        """
        # Store the bounding box
        self.objects[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """
        Deregisters an object that has been lost for too long.
        
        Parameters:
        objectID (int): The ID of the object to deregister.
        """
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        The core update logic for the tracker.
        Accepts bounding box rectangles from the detector for the current frame,
        associates them with existing objects, and returns the updated set of bounding boxes.
        
        Parameters:
        rects (list of tuples): A list of (startX, startY, endX, endY) bounding boxes from the detector.
        
        Returns:
        OrderedDict: The dictionary of currently tracked objects {objectID: (startX, startY, endX, endY)}.
        """
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(rects[i])
        else:
            objectIDs = list(self.objects.keys())
            
            # Calculate centroids for our existing tracked objects
            object_rects = list(self.objects.values())
            objectCentroids = np.array([ (int((r[0] + r[2]) / 2.0), int((r[1] + r[3]) / 2.0)) for r in object_rects])

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                # Update the object with the new bounding box
                self.objects[objectID] = rects[col]
                self.disappeared[objectID] = 0
                
                usedRows.add(row)
                usedCols.add(col)

            # Get the indices of rows and columns we did NOT use
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            # If there are more existing objects than new detections,
            # some objects may have disappeared. Mark them.
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # If there are new detections that were not matched,
            # register them as new objects.
            for col in unusedCols:
                self.register(rects[col])
        
        return self.objects


def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            exit()

def run_object_tracker():
    """
    Main function to run the tracking-by-detection pipeline.
    Initializes YOLO, the CentroidTracker, and a video stream,
    then runs the detection and tracking loop.
    """
    # --- 1. Download and Load Model ---
    yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    yolo_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    weights_file = "yolov3.weights"
    cfg_file = "yolov3.cfg"
    names_file = "coco.names"
    download_file(yolo_weights_url, weights_file)
    download_file(yolo_cfg_url, cfg_file)
    download_file(coco_names_url, names_file)

    with open(names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNet(weights_file, cfg_file)
    output_layer_names = net.getUnconnectedOutLayersNames()

    # --- 2. Initialize Tracker and Video Stream ---
    # Instantiate our centroid tracker
    ct = CentroidTracker()
    
    print("Initializing webcam...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- 3. Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        (H, W) = frame.shape[:2]
        
        # --- 4. Run Detection ---
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layer_names)
        
        # This list will hold the bounding boxes for the tracker
        rects = []

        # Post-process detections (similar to previous labs, but we only keep person boxes)
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # We will only try to track 'person' objects for this example
                if classes[class_id] == "person" and confidence > 0.95:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # Add the bounding box (startX, startY, endX, endY)
                    rects.append((x, y, x + int(width), y + int(height)))
        
        # --- 5. Update Tracker ---
        # Pass the list of bounding boxes to our tracker
        objects = ct.update(rects)

        # --- 6. Visualize Tracking Results ---
        # Loop over the tracked objects returned by the tracker
        for (objectID, rect) in objects.items():
            # Unpack the bounding box coordinates from our tracker
            (startX, startY, endX, endY) = rect
            
            # Draw the bounding box rectangle for the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Create the label text and draw it above the bounding box
            label = f"Person ID: {objectID}"
            y_pos = startY - 10 if startY > 20 else startY + 20
            cv2.putText(frame, label, (startX, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate the centroid and draw it
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # Put the ID label near the centroid point
            centroid_text = f"{objectID}"
            cv2.putText(frame, centroid_text, (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # Display the output
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_object_tracker()
