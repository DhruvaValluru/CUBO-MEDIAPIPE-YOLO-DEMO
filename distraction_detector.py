"""
Distraction Detector
====================
Uses your webcam to detect if you are distracted by:
  - Holding a phone AND looking toward it  (phone distraction)

A full MediaPipe face mesh overlay is drawn on your face.

Dependencies:
  - MediaPipe Face Mesh   (head-pose estimation + landmark overlay)
  - YOLOv8 / Ultralytics  (phone detection via COCO "cell phone" class)
  - OpenCV                 (camera & display)
  - Flask                  (web streaming)

Run:
  python distraction_detector.py
Open http://localhost:5000 in your browser.
"""

import cv2
import math
import time
import threading
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# ────────────────────────────── constants ──────────────────────────────

# COCO class index for "cell phone" in YOLOv8
PHONE_CLASS_ID = 67

# How close (in degrees) your face direction must be to the phone location
# to count as "looking at it". Lower = stricter, higher = more lenient.
GAZE_TO_PHONE_ANGLE_THRESHOLD = 25

# Detection confidence thresholds
PHONE_CONF_THRESHOLD = 0.45
FACE_DETECTION_CONF = 0.5
FACE_TRACKING_CONF = 0.5

# Colours (BGR)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specs for the face mesh overlay
MESH_TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
MESH_CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
MESH_IRISES_STYLE = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=1)

# Smoothing: how many consecutive "distracted" frames before we flag
DISTRACTION_FRAME_BUFFER = 8

# ────────────────────────────── helpers ────────────────────────────────


def rotation_matrix_to_euler(R):
    """Convert a 3x3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Use 6 canonical face landmarks to solve a PnP problem and return
    (pitch, yaw, roll) in degrees.

    Landmark indices (MediaPipe Face Mesh 468-point):
        1   – nose tip
        33  – right eye outer corner
        263 – left  eye outer corner
        61  – right mouth corner
        291 – left  mouth corner
        199 – chin
    """
    # 3-D model points (generic face model, arbitrary units)
    model_points = np.array([
        (0.0,    0.0,    0.0),      # nose tip
        (-30.0, -125.0, -30.0),     # chin
        (-225.0, 170.0, -135.0),    # left  eye outer
        (225.0,  170.0, -135.0),    # right eye outer
        (-150.0, -150.0, -125.0),   # left  mouth corner
        (150.0,  -150.0, -125.0),   # right mouth corner
    ], dtype=np.float64)

    # Corresponding 2-D image points from detected landmarks
    indices = [1, 199, 263, 33, 291, 61]
    image_points = np.array([
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
        for i in indices
    ], dtype=np.float64)

    # Camera internals (approximation)
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler(rmat)
    return pitch, yaw, roll


def angle_between_gaze_and_phone(pitch, yaw, face_cx, face_cy, phone_box, frame_w, frame_h):
    """
    Compute the angular difference between where the face is pointing
    (from head-pose pitch/yaw) and the direction from the face toward the
    phone center. Returns the angle in degrees – small = looking at phone.
    """
    # Direction the face is pointing (unit vector from pitch/yaw)
    yaw_r = math.radians(yaw)
    pitch_r = math.radians(pitch)
    gaze_dx = -math.sin(yaw_r)
    gaze_dy = math.sin(pitch_r)

    # Direction from face center to phone center (in image coords, normalised)
    px1, py1, px2, py2 = phone_box
    phone_cx = (px1 + px2) / 2
    phone_cy = (py1 + py2) / 2
    dir_dx = (phone_cx - face_cx) / frame_w
    dir_dy = (phone_cy - face_cy) / frame_h

    # Angle between the two 2-D vectors
    dot = gaze_dx * dir_dx + gaze_dy * dir_dy
    mag1 = math.sqrt(gaze_dx ** 2 + gaze_dy ** 2) + 1e-9
    mag2 = math.sqrt(dir_dx ** 2 + dir_dy ** 2) + 1e-9
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def draw_label(frame, text, origin, color, bg=True):
    """Draw a text label with optional background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    if bg:
        cv2.rectangle(frame, (x - 4, y - th - 8), (x + tw + 4, y + baseline + 4), BLACK, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ────────────────────────────── Flask app ──────────────────────────────

app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Distraction Detector</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0f0f0f;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 24px;
    }
    h1 {
      font-size: 1.6rem;
      font-weight: 600;
      margin-bottom: 16px;
      letter-spacing: -0.02em;
    }
    .stream-container {
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      max-width: 960px;
      width: 100%;
    }
    .stream-container img {
      width: 100%;
      display: block;
    }
    p {
      margin-top: 16px;
      color: #888;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>Distraction Detector</h1>
  <div class="stream-container">
    <img src="/video_feed" alt="Live webcam feed">
  </div>
  <p>Detecting phone usage via YOLOv8 + MediaPipe Face Mesh</p>
</body>
</html>"""

# Shared state for the latest JPEG frame
output_frame = None
frame_lock = threading.Lock()


def detection_loop():
    """Background thread: captures webcam, runs detection, writes JPEG frames."""
    global output_frame

    print("[INFO] Loading YOLOv8 model (first run downloads ~6 MB)...")
    yolo = YOLO("yolov8n.pt")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=FACE_DETECTION_CONF,
        min_tracking_confidence=FACE_TRACKING_CONF,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    distraction_counter = 0
    fps_time = time.time()

    print("[INFO] Detection loop running.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Phone detection (YOLOv8) ──
        yolo_results = yolo.predict(rgb, conf=PHONE_CONF_THRESHOLD, verbose=False)[0]
        phone_detected = False
        phone_boxes = []

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == PHONE_CLASS_ID:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                phone_boxes.append((x1, y1, x2, y2))
                box_w = x2 - x1
                inset = int(box_w * 0.30)
                dx1, dx2 = x1 + inset, x2 - inset
                cv2.rectangle(frame, (dx1, y1), (dx2, y2), YELLOW, 2)
                draw_label(frame, f"Phone {conf:.0%}", (dx1, y1 - 10), YELLOW)

        # ── Face mesh / head-pose estimation ──
        face_results = face_mesh.process(rgb)
        pitch, yaw, roll = 0.0, 0.0, 0.0
        face_found = False
        face_cx = w / 2
        face_cy = h / 2

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            face_found = True

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=MESH_TESSELATION_STYLE,
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=MESH_CONTOUR_STYLE,
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=MESH_IRISES_STYLE,
            )

            nose = face_landmarks.landmark[1]
            face_cx = nose.x * w
            face_cy = nose.y * h

            pose = estimate_head_pose(face_landmarks.landmark, w, h)
            if pose is not None:
                pitch, yaw, roll = pose

        # ── Distraction logic ──
        distracted = False
        reason = ""
        gaze_angle = None

        if face_found and phone_detected:
            for pbox in phone_boxes:
                angle = angle_between_gaze_and_phone(
                    pitch, yaw, face_cx, face_cy, pbox, w, h
                )
                gaze_angle = angle
                if angle < GAZE_TO_PHONE_ANGLE_THRESHOLD:
                    distracted = True
                    reason = f"Looking at phone (angle {angle:.0f} deg)"
                    break

        if distracted:
            distraction_counter = min(distraction_counter + 1, DISTRACTION_FRAME_BUFFER + 5)
        else:
            distraction_counter = max(distraction_counter - 1, 0)

        is_distracted = distraction_counter >= DISTRACTION_FRAME_BUFFER

        # ── Draw overlay ──
        if is_distracted:
            cv2.rectangle(frame, (0, 0), (w, 50), RED, -1)
            draw_label(frame, f"DISTRACTED: {reason}", (10, 35), WHITE, bg=False)
        else:
            cv2.rectangle(frame, (0, 0), (w, 50), GREEN, -1)
            draw_label(frame, "FOCUSED", (10, 35), WHITE, bg=False)

        panel_y = 70
        draw_label(frame, f"Yaw:   {yaw:+6.1f} deg", (10, panel_y), WHITE)
        draw_label(frame, f"Pitch: {pitch:+6.1f} deg", (10, panel_y + 30), WHITE)
        draw_label(frame, f"Roll:  {roll:+6.1f} deg", (10, panel_y + 60), WHITE)
        draw_label(frame, f"Phone: {'YES' if phone_detected else 'no'}", (10, panel_y + 90),
                   YELLOW if phone_detected else GREEN)
        draw_label(frame, f"Face:  {'YES' if face_found else 'no'}", (10, panel_y + 120),
                   GREEN if face_found else RED)
        if gaze_angle is not None:
            angle_color = RED if gaze_angle < GAZE_TO_PHONE_ANGLE_THRESHOLD else GREEN
            draw_label(frame, f"Gaze->Phone: {gaze_angle:.1f} deg (thresh {GAZE_TO_PHONE_ANGLE_THRESHOLD})",
                       (10, panel_y + 150), angle_color)

        now = time.time()
        fps_display = 1.0 / max(now - fps_time, 1e-9)
        fps_time = now
        draw_label(frame, f"FPS: {fps_display:.0f}", (w - 140, panel_y), WHITE)

        guide_y = h - 20
        draw_label(frame, f"Gaze-to-phone angle thresh: <{GAZE_TO_PHONE_ANGLE_THRESHOLD} deg = distracted",
                   (10, guide_y), (180, 180, 180))

        # Encode frame as JPEG and store for streaming
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            output_frame = jpeg.tobytes()

    cap.release()
    face_mesh.close()
    print("[INFO] Detection loop stopped.")


def generate_mjpeg():
    """Yield MJPEG frames for the /video_feed endpoint."""
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame_bytes = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("[INFO] Open http://localhost:8080 in your browser.")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
