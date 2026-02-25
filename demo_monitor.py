"""
Demo monitor - Basler GigE camera + YOLO can detection
Simple single-threaded loop: capture -> detect -> display
Controls: 'q' quit, 's' screenshot, 'r' record, 'c' clear stats
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
import time
import os
from datetime import datetime
from collections import deque


# --- CONFIGURATION ---
# MODEL_PATH = "runs/detect/train3/weights/best.pt"
MODEL_PATH = "best_model/best.pt"
CLASS_NAMES = ["0_5L", "0_33L", "0_25L"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR
CONFIDENCE = 0.5
PANEL_WIDTH = 300


def init_camera():
    """Initialize Basler GigE camera in SingleFrame mode."""
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")

    camera.AcquisitionMode.SetValue("SingleFrame")
    camera.TriggerMode.SetValue("Off")
    camera.PixelFormat.SetValue("Mono8")
    camera.GevSCPSPacketSize.SetValue(1500)
    camera.GevSCPD.SetValue(3000)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return camera, converter


def grab_frame(camera, converter):
    """Grab a single frame from the Basler camera. Returns BGR numpy array or None."""
    try:
        grab = camera.GrabOne(5000)
    except Exception as e:
        print(f"Grab error: {e}")
        return None

    if not grab.GrabSucceeded():
        grab.Release()
        return None

    frame = converter.Convert(grab).GetArray()
    grab.Release()
    return frame


def detect(model, frame):
    """Run YOLO inference. Returns annotated frame and per-class detection counts."""
    results = model(frame, verbose=False, conf=CONFIDENCE, half=True)
    annotated = frame.copy()
    counts = {i: 0 for i in range(len(CLASS_NAMES))}

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls < len(CLASS_NAMES):
                counts[cls] += 1
                color = COLORS[cls]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES[cls]} {conf:.0%}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated, counts


def build_display(frame, counts, fps, recording):
    """Combine camera frame with a side info panel."""
    h, w = frame.shape[:2]
    max_w = 980
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
        h, w = frame.shape[:2]

    panel_h = max(h, 400)
    panel = np.full((panel_h, PANEL_WIDTH, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30

    # Title
    cv2.putText(panel, "CAN MONITOR", (10, y), font, 0.8, (0, 255, 255), 2)
    cv2.line(panel, (10, y + 5), (PANEL_WIDTH - 10, y + 5), (0, 255, 255), 2)
    y += 40

    # Time
    cv2.putText(panel, datetime.now().strftime("%H:%M:%S"), (10, y), font, 0.5, (255, 255, 255), 1)
    y += 30

    # Recording
    rec_color = (0, 0, 255) if recording else (128, 128, 128)
    cv2.putText(panel, "REC" if recording else "STANDBY", (10, y), font, 0.5, rec_color, 1)
    y += 35

    # FPS
    cv2.putText(panel, "PERFORMANCE", (10, y), font, 0.6, (255, 200, 0), 2)
    y += 25
    cv2.putText(panel, f"FPS: {fps:.1f}", (10, y), font, 0.5, (0, 255, 0), 1)
    y += 35

    # Detections
    cv2.putText(panel, "DETECTIONS", (10, y), font, 0.6, (255, 200, 0), 2)
    y += 25
    cv2.putText(panel, f"Total: {sum(counts.values())}", (10, y), font, 0.5, (0, 255, 255), 1)
    y += 25
    for i, name in enumerate(CLASS_NAMES):
        c = counts.get(i, 0)
        color = COLORS[i] if c > 0 else (128, 128, 128)
        cv2.putText(panel, f"  {name}: {c}", (10, y), font, 0.45, color, 1)
        y += 22

    # Combine
    out_h = max(h, panel_h)
    combined = np.zeros((out_h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    combined[:h, :w] = frame
    combined[:panel_h, w:] = panel
    return combined


def main():
    print("Can Detector Monitor")
    print("=" * 40)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to("cuda")
    print("Model loaded (GPU)")

    # Init camera
    camera, converter = init_camera()
    print("Camera ready")
    print("Controls: 'q' quit | 's' screenshot | 'r' record | 'c' clear stats")

    # State
    recording = False
    video_writer = None
    fps_history = deque(maxlen=30)
    frame_count = 0
    last_fps_time = time.time()

    os.makedirs("demo_screenshots", exist_ok=True)
    os.makedirs("demo_recordings", exist_ok=True)

    try:
        while True:
            # 1. Capture
            frame = grab_frame(camera, converter)
            if frame is None:
                continue

            # 2. Detect
            annotated, counts = detect(model, frame)

            # 3. FPS
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps_history.append(frame_count / (now - last_fps_time))
                frame_count = 0
                last_fps_time = now
            fps = fps_history[-1] if fps_history else 0

            # 4. Display
            display = build_display(annotated, counts, fps, recording)

            if recording and video_writer is not None:
                video_writer.write(display)

            cv2.imshow("Can Detector Monitor", display)

            # 5. Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"demo_screenshots/demo_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(fname, display)
                print(f"Screenshot saved: {fname}")
            elif key == ord("r"):
                if not recording:
                    fname = f"demo_recordings/rec_{datetime.now():%Y%m%d_%H%M%S}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    video_writer = cv2.VideoWriter(fname, fourcc, 15,
                                                   (display.shape[1], display.shape[0]))
                    recording = True
                    print(f"Recording started: {fname}")
                else:
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    recording = False
                    print("Recording stopped")
            elif key == ord("c"):
                fps_history.clear()
                frame_count = 0
                last_fps_time = time.time()
                print("Stats cleared")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        camera.Close()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()