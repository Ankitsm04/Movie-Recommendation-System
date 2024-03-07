import cv2
import numpy as np

video = 'cars.mp4'
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first frame
ret, old_frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Create random points for tracking
p0 = cv2.goodFeaturesToTrack(old_frame, maxCorners=100, qualityLevel=0.3, minDistance=7)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Convert frames to gray scale
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)

    # Display result
    cv2.imshow('Optical Flow', img)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Break the loop with the 'Esc' key
    if cv2.waitKey(30) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
