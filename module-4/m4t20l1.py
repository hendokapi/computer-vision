
# Project Kickoff - The Baseline Tracker
# This script implements the baseline for our tracking project: a Centroid Tracker.
# It reads pre-computed detections from a file (in MOTChallenge format),
# associates them across frames using the Centroid Tracker, and visualizes the results.
# The tracker's output is saved to a file for later evaluation.

# Note: This script requires 'scipy'. Install with: pip install scipy
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import os
import requests
import zipfile
import configparser

# --- CentroidTracker Class (from previous lesson) ---
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
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
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])
        
        return self.objects

# --- Helper Functions for Data Handling ---
def download_and_unzip(url, save_path, extract_path):
    """Downloads and unzips a file if the extracted directory doesn't exist."""
    if os.path.exists(extract_path):
        print(f"Dataset already exists at {extract_path}")
        return

    print(f"Downloading and extracting dataset to {extract_path}...")
    if not os.path.exists(save_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {save_path}: {e}")
            return

    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

def read_detections(det_file_path):
    """Reads a MOTChallenge detection file and groups detections by frame."""
    detections = {}
    with open(det_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            frame_id = int(parts[0])
            # We don't use the given ID, our tracker will assign its own
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            
            # Filter detections by confidence
            if conf < 0.5:
                continue
                
            x2 = x1 + w
            y2 = y1 + h
            
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append((x1, y1, x2, y2))
    return detections

# --- Main Project Logic ---
def run_baseline_tracker():
    """
    Main function to run the baseline tracking project.
    Downloads data, reads detections, runs the CentroidTracker, and saves the output.
    """
    # --- 1. Download and Prepare Dataset ---
    # Download manually from https://www.kaggle.com/datasets/takshmandar/mot16-dataset
    # then extract train/MOT16-02 in the MOT16-02 directory under the root of the project

    # dataset_url = "https://data.vision.ee.ethz.ch/cvl/MOT16/MOT16-02.zip"
    # zip_path = "MOT16-02.zip"
    sequence_path = "MOT16-02"
    # download_and_unzip(dataset_url, zip_path, sequence_path)

    # --- 2. Read Sequence Information and Detections ---
    seqinfo_path = os.path.join(sequence_path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    seq_info = config['Sequence']
    
    imDir = os.path.join(sequence_path, seq_info['imDir'])
    frameRate = int(seq_info['frameRate'])
    seqLength = int(seq_info['seqLength'])
    imWidth = int(seq_info['imWidth'])
    imHeight = int(seq_info['imHeight'])
    imExt = seq_info['imExt']

    det_file = os.path.join(sequence_path, 'det', 'det.txt')
    detections_by_frame = read_detections(det_file)

    # --- 3. Initialize Tracker and Run Tracking Loop ---
    ct = CentroidTracker(maxDisappeared=int(frameRate * 1.5)) # Allow objects to be lost for 1.5 seconds
    tracker_output = [] # To store results for saving

    print("Processing sequence...")
    for frame_num in range(1, seqLength + 1):
        # Get the path for the current frame image
        frame_path = os.path.join(imDir, f"{frame_num:06d}{imExt}")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue

        # Get the detections for the current frame
        rects = detections_by_frame.get(frame_num, [])
        
        # Update the tracker with the new detections
        objects = ct.update(rects)
        
        # --- Visualization and Output Formatting ---
        # Create a mapping from current tracked object IDs to the original detection boxes
        # This is a bit tricky since the tracker only stores centroids.
        # We'll re-calculate distances to associate back for visualization/saving.
        current_rects_map = {tuple(obj_rect):None for obj_rect in rects}
        if rects and objects:
            # Get centroids of current rects
            rect_centroids = np.array([((r[0]+r[2])/2, (r[1]+r[3])/2) for r in rects])
            # Get tracked object centroids
            obj_ids = list(objects.keys())
            obj_centroids = np.array(list(objects.values()))
            # Find the closest rect for each tracked object
            D = dist.cdist(obj_centroids, rect_centroids)
            # Greedily assign the closest rect to each object for this frame
            for i, obj_id in enumerate(obj_ids):
                if D.shape[1] > 0:
                    best_match_idx = np.argmin(D[i])
                    # Only associate if the distance is small (e.g., within the box size)
                    if D[i, best_match_idx] < (rects[best_match_idx][2] - rects[best_match_idx][0]):
                       matched_rect = rects[best_match_idx]
                       current_rects_map[tuple(matched_rect)] = obj_id

        # Loop over all detections for this frame to draw and save
        for (x1, y1, x2, y2) in rects:
            assigned_id = current_rects_map.get(tuple((x1,y1,x2,y2)))
            if assigned_id is not None:
                # This detection was associated with a track
                color = (hash(str(assigned_id)) % 255, hash(str(assigned_id)[::-1]) % 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                text = f"ID {assigned_id}"
                cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Format for saving: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
                bb_left = x1
                bb_top = y1
                bb_width = x2 - x1
                bb_height = y2 - y1
                tracker_output.append(f"{frame_num},{assigned_id},{bb_left},{bb_top},{bb_width},{bb_height},-1,-1,-1,-1")

        # Display progress
        cv2.imshow("Tracking - Baseline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 4. Save Results and Cleanup ---
    output_file = "ECTS4_Topic20_L1_output.txt"
    with open(output_file, 'w') as f:
        f.writelines(tracker_output)
    print(f"Tracker output saved to {output_file}")

    cv2.destroyAllWindows()

# --- Main execution block ---
if __name__ == "__main__":
    run_baseline_tracker()
