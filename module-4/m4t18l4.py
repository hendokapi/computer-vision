
# Motion Analysis Lab
# This script demonstrates two core techniques for video motion analysis in real-time:
# 1. Motion Detection using an adaptive background subtractor (MOG2).
# 2. Motion Estimation using sparse optical flow (Lucas-Kanade).
# Both techniques are run simultaneously on a live webcam feed for comparison.

import cv2
import numpy as np

def run_motion_analysis():
    """
    Main function to run real-time motion analysis from a webcam.
    It initializes a background subtractor and an optical flow tracker,
    processes each frame, and visualizes the results of both techniques.
    """
    # --- 1. Initialize Webcam ---
    print("Initializing webcam...")
    # Open the default webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam initialized.")

    # --- 2. Setup for Background Subtraction ---
    # Create a MOG2 background subtractor object. This will learn the background over time.
    # history: Number of past frames used to build the model.
    # varThreshold: Threshold on the squared Mahalanobis distance to classify a pixel as foreground.
    # detectShadows: If true, the algorithm will detect and mark shadows (gray values in the mask).
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # Create a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- 3. Setup for Sparse Optical Flow (Lucas-Kanade) ---
    # Read the first frame to initialize the tracker
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame for optical flow setup.")
        cap.release()
        return
        
    # Convert the first frame to grayscale
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters for ShiTomasi corner detection (finding initial features to track)
    # maxCorners: Maximum number of corners to return.
    # qualityLevel: Minimal accepted quality of image corners.
    # minDistance: Minimum possible Euclidean distance between the returned corners.
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Find initial strong corners in the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Parameters for Lucas-Kanade optical flow
    # winSize: Size of the search window at each pyramid level.
    # maxLevel: 0-based maximal pyramid level number.
    # criteria: Termination criteria for the iterative search algorithm.
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create a mask image for drawing the optical flow tracks
    mask = np.zeros_like(old_frame)
    
    # --- 4. Main Processing Loop ---
    while True:
        # Read a new frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # --- 5. Background Subtraction Processing ---
        # Apply the background subtractor to the current frame to get the foreground mask
        fgMask = backSub.apply(frame)
        
        # Clean up the mask using morphological operations
        # Opening removes small noise (white dots)
        fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        # Closing fills small holes in the foreground objects
        fgMask_cleaned = cv2.morphologyEx(fgMask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of the moving objects in the cleaned mask
        contours, _ = cv2.findContours(fgMask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around detected moving objects on the original frame
        for contour in contours:
            # Filter out small contours that are likely noise
            if cv2.contourArea(contour) > 500:
                # Get the bounding box for the contour
                (x, y, w, h) = cv2.boundingRect(contour)
                # Draw the rectangle on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- 6. Optical Flow Processing ---
        # Convert the current frame to grayscale for optical flow calculation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow from the previous frame (old_gray) to the current frame (frame_gray)
        # p1 will contain the new positions of the tracked points (p0)
        # st (status) is an output vector; st[i] is 1 if the flow for feature i has been found, else 0.
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points (points that were successfully tracked)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        else:
            good_new, good_old = [], []

        # Draw the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Get the coordinates as integers
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            # Draw a line from the old point to the new point on the mask
            mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
            # Draw a circle at the new position on the original frame
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
        
        # Combine the frame with the optical flow mask
        img = cv2.add(frame, mask)

        # Update the previous frame and previous points for the next iteration
        old_gray = frame_gray.copy()
        # Update points to track, but only if there are any good points left
        p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # If we run out of points, re-detect them
        if p0 is None or len(p0) < 10:
             p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # --- 7. Display Results ---
        # Show the final combined output
        cv2.imshow("Motion Analysis (Detection & Flow)", img)
        # Show the raw and cleaned foreground masks for comparison
        cv2.imshow("Raw Foreground Mask", fgMask)
        cv2.imshow("Cleaned Foreground Mask", fgMask_cleaned)
        
        # Check for 'q' key press to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # --- 8. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

# --- Main execution block ---
if __name__ == "__main__":
    run_motion_analysis()
