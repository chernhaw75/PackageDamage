import cv2
from ultralytics import YOLO
import numpy as np
import time
import os

# 1. Load Model
model = YOLO("best.pt") 

# 2. Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a folder for false negative evidence
if not os.path.exists('false_negatives'):
    os.makedirs('false_negatives')

damaged_counter = 0
is_counting = False
prev_time = 0 

print("System Active. Press 'S' to save a screenshot for the model team.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS for loop verification
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    damaged_conf = 0.0
    undamaged_conf = 0.0

    # Inference (Set conf=0.001 to catch EVERYTHING for the model team)
    results = model(frame, stream=True, verbose=False, conf=0.001)

    for result in results:
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for cls_idx, score in zip(classes, scores):
                cls_name = model.names[int(cls_idx)].lower()
                if "damaged" in cls_name:
                    damaged_conf = max(damaged_conf, float(score))
                elif "undamaged" in cls_name:
                    undamaged_conf = max(undamaged_conf, float(score))

    # --- UI Logic ---
    # Threshold for the "Alert" bar remains at 0.5, but numbers show 0.000%
    if damaged_conf > 0.5:
        status_text, bg_color = "ALARM: DAMAGE", (0, 0, 255)
        if not is_counting:
            damaged_counter += 1
            is_counting = True
    elif undamaged_conf > 0.5:
        status_text, bg_color = "SYSTEM: CLEAR", (0, 255, 0)
        is_counting = False
    else:
        status_text, bg_color = "SYSTEM: MONITORING", (60, 60, 60)
        is_counting = False

    # --- Draw HUD ---
    # Top Bar
    cv2.rectangle(frame, (0, 0), (640, 50), bg_color, -1)
    cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (540, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bottom HUD (Black Overlay)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 430), (640, 480), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Display raw confidence (4 decimal places to show even tiny model noise)
    # This proves the screen is not frozen because these numbers jitter constantly
    cv2.putText(frame, f"UNDAMAGED: {undamaged_conf:.4%}", (20, 465), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"DAMAGED: {damaged_conf:.4%}", (330, 465), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Heartbeat indicator
    if int(time.time() * 5) % 2 == 0:
        cv2.circle(frame, (620, 455), 5, (0, 255, 255), -1)

    cv2.imshow("Jetson Model Feedback Tool", frame)

    key = cv2.waitKey(1) & 0xFF
    # Save a screenshot if the user presses 's' to report a false negative
    if key == ord('s') or key == ord('S'):
        fname = f"false_negatives/FN_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Screenshot saved for model team: {fname}")
    elif key in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
