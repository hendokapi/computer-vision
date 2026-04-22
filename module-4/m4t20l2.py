
# Improving Association with a Kalman Filter (SORT)
# This script incrementally improves upon the baseline tracker from L1.
# It replaces the simple CentroidTracker with a more robust SORT-based tracker.
# Key changes:
# 1. Each tracked object is now managed by a Kalman Filter to predict its motion.
# 2. Association is no longer based on centroid distance but on the Intersection over Union (IoU) of bounding boxes.
# 3. The Hungarian algorithm is used for optimal assignment, minimizing the total association cost.

# Note: This script requires 'scipy'. Install with: pip install scipy
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import os
import requests
import zipfile
import configparser

# --- Kalman Filter and Tracker Classes ---

def iou(bb_test, bb_gt):
    """Computes IUO between two bboxes in the form [x1,y1,x2,y2]"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
       the aspect ratio"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
       [x1,y1,x2,y2] where x1,y1 is the top-left and x2,y2 is the bottom-right"""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
  """This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    #define constant velocity model
    self.kf = cv2.KalmanFilter(7, 4)
    self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]],np.float32)
    self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],np.float32)
    self.kf.processNoiseCov = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],np.float32) * 0.03

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.kf.statePost = np.zeros((7, 1), np.float32)
    self.kf.statePost[:4] = convert_bbox_to_z(bbox)

  def update(self,bbox):
    """Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.correct(convert_bbox_to_z(bbox))

  def predict(self):
    """Advances the state vector and returns the predicted bbox estimate.
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.statePost[:4,0]))
    return self.history[-1]

  def get_state(self):
    """Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.statePost[:4,0])

class SortTracker(object):
  """This class represents the internal state of individual tracked objects observed as bbox.
  """
  def __init__(self, max_age=1, min_hits=3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d, t] = iou(det, trk)
    
    # Use the Hungarian algorithm (linear_sum_assignment) for optimal matching
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)

    unmatched_detections = []
    for d, det in enumerate(dets):
      if(d not in matched_indices[:,0]):
        unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trks):
      if(t not in matched_indices[:,1]):
        unmatched_trackers.append(t)
    
    #update matched trackers with assigned detections
    for m in matched_indices:
      self.trackers[m[1]].update(dets[m[0], :])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_detections:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

# --- Helper Functions for Data Handling (same as L1) ---
def download_and_unzip(url, save_path, extract_path): 
    if os.path.exists(extract_path): return
    print(f"Downloading and extracting dataset...")
    if not os.path.exists(save_path):
        try:
            r = requests.get(url, stream=True); r.raise_for_status()
            with open(save_path, 'wb') as f: [f.write(c) for c in r.iter_content(chunk_size=8192)]
        except Exception as e: print(f"Error downloading: {e}"); return
    with zipfile.ZipFile(save_path, 'r') as z: z.extractall(os.path.dirname(extract_path))

def read_detections(det_file_path):
    detections = {}
    with open(det_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id, _, x1, y1, w, h, conf, _, _, _ = [float(p) for p in parts]
            if conf < 0.5: continue
            if int(frame_id) not in detections: detections[int(frame_id)] = []
            detections[int(frame_id)].append([x1, y1, x1 + w, y1 + h, conf])
    return detections

# --- Main Project Logic ---
def run_sort_tracker():
    # Download manually from https://www.kaggle.com/datasets/takshmandar/mot16-dataset
    # then extract train/MOT16-02 in the MOT16-02 directory under the root of the project

    # dataset_url = "https://data.vision.ee.ethz.ch/cvl/MOT16/MOT16-02.zip"
    # zip_path = "MOT16-02.zip"
    sequence_path = "MOT16-02"
    # download_and_unzip(dataset_url, zip_path, sequence_path)

    seqinfo_path = os.path.join(sequence_path, 'seqinfo.ini')
    config = configparser.ConfigParser(); config.read(seqinfo_path)
    seq_info = config['Sequence']
    imDir, frameRate, seqLength = os.path.join(sequence_path, seq_info['imDir']), int(seq_info['frameRate']), int(seq_info['seqLength'])
    imExt = seq_info['imExt']

    det_file = os.path.join(sequence_path, 'det', 'det.txt')
    detections_by_frame = read_detections(det_file)

    # Initialize the SORT tracker
    sort_tracker = SortTracker(max_age=int(frameRate/2), min_hits=3)
    tracker_output = []

    print("Processing sequence with SORT tracker...")
    for frame_num in range(1, seqLength + 1):
        frame_path = os.path.join(imDir, f"{frame_num:06d}{imExt}")
        frame = cv2.imread(frame_path)
        if frame is None: continue

        # Get detections for the current frame and convert to numpy array
        dets = np.array(detections_by_frame.get(frame_num, []))
        if len(dets) == 0: dets = np.empty((0, 5))
        
        # Update the tracker
        tracked_objects = sort_tracker.update(dets)

        # Visualization and Output Formatting
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            color = (hash(str(obj_id)) % 255, hash(str(obj_id)[::-1]) % 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            bb_left, bb_top = x1, y1
            bb_width, bb_height = (x2 - x1), (y2 - y1)
            tracker_output.append(f"{frame_num},{obj_id},{bb_left},{bb_top},{bb_width},{bb_height},-1,-1,-1,-1")

        cv2.imshow("Tracking - SORT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    output_file = "ECTS4_Topic20_L2_output.txt"
    with open(output_file, 'w') as f: f.writelines(tracker_output)
    print(f"Tracker output saved to {output_file}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_sort_tracker()
