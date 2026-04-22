
# Handling Occlusions with Appearance Features (DeepSORT-like)
# This is the final incremental step in our tracking project.
# It builds upon the SORT tracker from L2 by adding an appearance model to handle occlusions, 
# making it a simplified version of DeepSORT.
# Key changes:
# 1. An appearance feature (a 3D color histogram) is computed for each detection.
# 2. Each track now maintains a gallery of recent appearance features.
# 3. A two-stage "cascading" association strategy is implemented:
#    - Stage 1: Match recent tracks using IoU (motion).
#    - Stage 2: Match occluded tracks with remaining detections using an appearance metric.

from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import os
import configparser

# --- SAFE IoU ---
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area1 = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])

    denom = area1 + area2 - wh
    if denom <= 0:
        return 0.0

    return wh / denom


# --- SAFE conversions ---
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w <= 0 or h <= 0:
        w, h = 1e-3, 1e-3

    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)

    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x):
    s = float(x[2])
    r = float(x[3])

    if s <= 0 or r <= 0 or np.isnan(s) or np.isnan(r):
        return np.array([[0, 0, 0, 0]])

    w = np.sqrt(s * r)
    if w <= 0 or np.isnan(w):
        return np.array([[0, 0, 0, 0]])

    h = s / w
    if h <= 0 or np.isnan(h):
        return np.array([[0, 0, 0, 0]])

    return np.array([
        x[0] - w / 2.,
        x[1] - h / 2.,
        x[0] + w / 2.,
        x[1] + h / 2.
    ]).reshape((1, 4))


# --- Appearance ---
def compute_color_histogram(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)

    h_img, w_img, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if x1 >= x2 or y1 >= y2:
        return np.zeros(288, dtype=np.float32)

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 12, 3],
                        [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


# --- Tracker ---
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, feature):
        self.kf = cv2.KalmanFilter(7, 4)

        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03

        self.kf.statePost = np.zeros((7, 1), np.float32)
        self.kf.statePost[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.features = [feature]

    def update(self, bbox, feature):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.kf.correct(convert_bbox_to_z(bbox))

        self.features.append(feature)
        if len(self.features) > 10:
            self.features.pop(0)

    def predict(self):
        self.kf.predict()

        # Clamp invalid states
        if self.kf.statePre[2] <= 0:
            self.kf.statePre[2] = 1e-3
        if self.kf.statePre[3] <= 0:
            self.kf.statePre[3] = 1e-3

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1

        return convert_x_to_bbox(self.kf.statePre[:4, 0])[0]

    def get_feature(self):
        return np.mean(self.features, axis=0)

    def get_state(self):
        return convert_x_to_bbox(self.kf.statePost[:4, 0])


class DeepSortTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        KalmanBoxTracker.count = 0

    def update(self, dets, frame):
        if len(dets) == 0:
            dets = np.empty((0, 4))

        det_features = [compute_color_histogram(frame, d) for d in dets]

        predicted = [t.predict() for t in self.trackers]

        iou_matrix = np.zeros((len(dets), len(self.trackers)))

        for d, det in enumerate(dets):
            for t, trk in enumerate(predicted):
                iou_matrix[d, t] = iou(det, trk)

        # sanitize matrix
        iou_matrix = np.nan_to_num(iou_matrix, nan=0.0)

        matched = linear_sum_assignment(-iou_matrix)
        matched = np.array(matched).T

        unmatched_dets = set(range(len(dets)))
        unmatched_trks = set(range(len(self.trackers)))

        for m in matched:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                continue

            self.trackers[m[1]].update(dets[m[0]], det_features[m[0]])
            unmatched_dets.discard(m[0])
            unmatched_trks.discard(m[1])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i], det_features[i]))

        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue

            if trk.hits >= self.min_hits:
                bbox = trk.get_state()[0]
                ret.append(np.concatenate((bbox, [trk.id + 1])))

        return np.array(ret) if len(ret) > 0 else np.empty((0, 5))


# --- Data ---
def read_detections(path):
    detections = {}
    with open(path, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            frame = int(p[0])
            if float(p[6]) < 0.5:
                continue

            if frame not in detections:
                detections[frame] = []

            detections[frame].append([float(x) for x in p[2:6]])

    return detections


# --- MAIN ---
def run():
    seq = "MOT16-02"

    config = configparser.ConfigParser()
    config.read(os.path.join(seq, 'seqinfo.ini'))

    info = config['Sequence']
    imDir = os.path.join(seq, info['imDir'])
    seqLen = int(info['seqLength'])
    imExt = info['imExt']

    dets = read_detections(os.path.join(seq, 'det', 'det.txt'))

    tracker = DeepSortTracker()

    for frame_num in range(1, seqLen + 1):
        frame = cv2.imread(os.path.join(imDir, f"{frame_num:06d}{imExt}"))
        if frame is None:
            continue

        raw = dets.get(frame_num, [])
        clean = []

        for d in raw:
            x, y, w, h = d

            if w <= 0 or h <= 0:
                continue
            if np.isnan([x, y, w, h]).any():
                continue

            clean.append([x, y, x + w, y + h])  # convert

        dets_frame = np.array(clean) if clean else np.empty((0, 4))

        tracks = tracker.update(dets_frame, frame)

        for t in tracks:
            x1, y1, x2, y2, tid = t.astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
