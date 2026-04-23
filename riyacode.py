# Hand Gesture Recognition System v3.0
# Compatible with Python 3.13 + mediapipe 0.10.33
# Uses the NEW MediaPipe Tasks API (no mp.solutions needed)
#
# Install: pip install mediapipe opencv-python numpy
# Run:     python riya_code_py313.py
#          python riya_code_py313.py --multi
#          python riya_code_py313.py --app volume
#          python riya_code_py313.py --app slides

import cv2
import numpy as np
import time
import argparse
import urllib.request
import os
from collections import deque, Counter
from enum import Enum

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ------------------------------------
#  MODEL DOWNLOAD
# ------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~5 MB), please wait...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
        print("")

# ------------------------------------
#  CONFIG
# ------------------------------------
class Config:
    HISTORY_SIZE       = 7
    MIN_DETECTION_CONF = 0.75
    MIN_PRESENCE_CONF  = 0.75
    MIN_TRACKING_CONF  = 0.75

    COLOR_PRIMARY  = (0, 220, 180)
    COLOR_WHITE    = (255, 255, 255)
    COLOR_BLACK    = (0, 0, 0)
    COLOR_DARK_BG  = (20, 20, 30)
    COLOR_LEFT     = (255, 140, 0)
    COLOR_RIGHT    = (0, 200, 255)

    GESTURE_ACTIONS = {
        0: "STOP",
        1: "POINT",
        2: "PEACE",
        3: "ROCK-ON",
        4: "FOUR",
        5: "START / GO",
    }


class AppMode(Enum):
    DISPLAY = "display"
    VOLUME  = "volume"
    SLIDES  = "slides"


# ------------------------------------
#  LANDMARK INDEX CONSTANTS
# ------------------------------------
THUMB_IP   = 3
THUMB_TIP  = 4
INDEX_PIP  = 6
INDEX_TIP  = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP   = 14
RING_TIP   = 16
PINKY_PIP  = 18
PINKY_TIP  = 20

FINGERTIP_IDS  = [INDEX_TIP,  MIDDLE_TIP,  RING_TIP,  PINKY_TIP]
FINGER_PIP_IDS = [INDEX_PIP,  MIDDLE_PIP,  RING_PIP,  PINKY_PIP]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ------------------------------------
#  GESTURE BUFFER
# ------------------------------------
class GestureBuffer:
    def __init__(self, size=Config.HISTORY_SIZE):
        self.buffer = deque(maxlen=size)

    def update(self, count):
        self.buffer.append(count)

    def get_stable_count(self):
        if not self.buffer:
            return 0
        return Counter(self.buffer).most_common(1)[0][0]

    def get_stability(self):
        if len(self.buffer) < 2:
            return 1.0
        top = Counter(self.buffer).most_common(1)[0][1]
        return top / len(self.buffer)


# ------------------------------------
#  FINGER COUNTING
# ------------------------------------
def count_fingers(landmarks, handedness_label):
    lm = landmarks
    finger_states = []

    tip_x = lm[THUMB_TIP].x
    ip_x  = lm[THUMB_IP].x
    if handedness_label == "Right":
        finger_states.append(tip_x < ip_x)
    else:
        finger_states.append(tip_x > ip_x)

    for tip_id, pip_id in zip(FINGERTIP_IDS, FINGER_PIP_IDS):
        finger_states.append(lm[tip_id].y < lm[pip_id].y)

    count    = sum(finger_states)
    z_spread = max(abs(lm[tip].z) for tip in FINGERTIP_IDS)
    conf     = round(max(0.5, 1.0 - min(z_spread * 3, 0.5)), 2)

    return count, finger_states, conf


# ------------------------------------
#  DRAWING HELPERS
# ------------------------------------
def draw_skeleton(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, clen=20):
    cv2.line(frame, (x1, y1), (x1+clen, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1+clen), color, thickness)
    cv2.line(frame, (x2, y1), (x2-clen, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1+clen), color, thickness)
    cv2.line(frame, (x1, y2), (x1+clen, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2-clen), color, thickness)
    cv2.line(frame, (x2, y2), (x2-clen, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2-clen), color, thickness)


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_app_action(frame, count, app_mode, w, h):
    vol_map = {0:"MUTE", 1:"Vol 20%", 2:"Vol 40%", 3:"Vol 60%", 4:"Vol 80%", 5:"Vol 100%"}
    sld_map = {0:"PAUSE", 1:"PREV", 2:"NEXT", 3:"ZOOM", 4:"BLANK", 5:"PRESENT"}
    mapping = vol_map if app_mode == AppMode.VOLUME else sld_map
    action  = mapping.get(count, "")
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2-130, h-90), (w//2+130, h-55), (30,30,50), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, ">> " + action, (w//2-100, h-65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, Config.COLOR_PRIMARY, 2, cv2.LINE_AA)


# ------------------------------------
#  MAIN UI DRAW
# ------------------------------------
def draw_ui(frame, detection_result, gesture_buffers, app_mode, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 70), Config.COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "FPS: " + str(round(fps, 1)), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, Config.COLOR_PRIMARY, 1, cv2.LINE_AA)
    cv2.putText(frame, "MODE: " + app_mode.value.upper(), (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Config.COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h-30), (w, h), Config.COLOR_DARK_BG, -1)
    cv2.putText(frame, "Q: Quit  |  M: Toggle Mode  |  Hand Gesture Recognition v3.0",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,160), 1, cv2.LINE_AA)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list     = detection_result.handedness

    if not hand_landmarks_list:
        cv2.putText(frame, "No hand detected", (w//2-110, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,100,200), 2, cv2.LINE_AA)
        cv2.putText(frame, "Show your hand to the camera", (w//2-160, h//2+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,120,160), 1, cv2.LINE_AA)
        return frame

    for hand_idx, (landmarks, handedness) in enumerate(
            zip(hand_landmarks_list, handedness_list)):

        label      = handedness[0].category_name
        hand_color = Config.COLOR_RIGHT if label == "Right" else Config.COLOR_LEFT

        draw_skeleton(frame, landmarks, hand_color)

        xs    = [int(lm.x * w) for lm in landmarks]
        ys    = [int(lm.y * h) for lm in landmarks]
        x_min = max(0, min(xs)-15)
        x_max = min(w, max(xs)+15)
        y_min = max(0, min(ys)-15)
        y_max = min(h, max(ys)+15)
        draw_corner_box(frame, x_min, y_min, x_max, y_max, hand_color)

        if hand_idx not in gesture_buffers:
            gesture_buffers[hand_idx] = GestureBuffer()

        raw_count, finger_states, confidence = count_fingers(landmarks, label)
        gesture_buffers[hand_idx].update(raw_count)
        stable_count = gesture_buffers[hand_idx].get_stable_count()
        stability    = gesture_buffers[hand_idx].get_stability()

        count_text = str(stable_count)
        ts = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 6)[0]
        tx = x_min + (x_max - x_min)//2 - ts[0]//2
        ty = max(y_min - 15, 80)
        cv2.putText(frame, count_text, (tx+3, ty+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, Config.COLOR_BLACK, 8, cv2.LINE_AA)
        cv2.putText(frame, count_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, hand_color, 6, cv2.LINE_AA)

        cv2.putText(frame, label + " Hand", (x_min, y_max+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 1, cv2.LINE_AA)

        bar_x = x_min
        bar_y = y_max + 35
        bar_w = x_max - x_min
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+8), (60,60,80), -1)
        conf_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+conf_w, bar_y+8),
                      lerp_color((0,80,220), (0,220,80), confidence), -1)
        cv2.putText(frame, "Conf: " + str(int(confidence*100)) + "%",
                    (bar_x, bar_y+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_WHITE, 1, cv2.LINE_AA)

        if stability >= 0.85:
            stab_label = "Stable"
            stab_color = (0, 220, 80)
        elif stability >= 0.6:
            stab_label = "Settling"
            stab_color = (0, 200, 220)
        else:
            stab_label = "Jitter"
            stab_color = (0, 80, 220)

        cv2.putText(frame, "* " + stab_label, (bar_x, bar_y+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, stab_color, 1, cv2.LINE_AA)

        dot_y   = y_max + 65
        dot_gap = 18
        dot_x0  = x_min + (bar_w - dot_gap*4)//2
        for fi, (state, lbl) in enumerate(zip(finger_states, ["T","I","M","R","P"])):
            dx        = dot_x0 + fi * dot_gap
            dot_color = (0, 220, 100) if state else (80, 80, 120)
            cv2.circle(frame, (dx, dot_y), 7, dot_color, -1)
            cv2.putText(frame, lbl, (dx-4, dot_y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, Config.COLOR_WHITE, 1, cv2.LINE_AA)

        action = Config.GESTURE_ACTIONS.get(stable_count, "")
        cv2.putText(frame, action, (x_min, dot_y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, Config.COLOR_PRIMARY, 1, cv2.LINE_AA)

        if app_mode != AppMode.DISPLAY:
            draw_app_action(frame, stable_count, app_mode, w, h)

    return frame


# ------------------------------------
#  MAIN LOOP
# ------------------------------------
def main(max_hands=1, app_mode=AppMode.DISPLAY):
    print("==========================================")
    print("  Hand Gesture Recognition System v3.0   ")
    print("  Python 3.13 Compatible Edition         ")
    print("  Press Q to quit | M to switch mode     ")
    print("==========================================")
    print("")

    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_hands,
        min_hand_detection_confidence=Config.MIN_DETECTION_CONF,
        min_hand_presence_confidence=Config.MIN_PRESENCE_CONF,
        min_tracking_confidence=Config.MIN_TRACKING_CONF,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gesture_buffers = {}
    modes     = list(AppMode)
    mode_idx  = modes.index(app_mode)
    fps_time  = time.time()
    fps       = 0.0
    frame_cnt = 0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_cnt += 1
            now = time.time()
            if now - fps_time >= 0.5:
                fps       = frame_cnt / (now - fps_time)
                frame_cnt = 0
                fps_time  = now

            frame = draw_ui(frame, result, gesture_buffers, modes[mode_idx], fps)

            cv2.imshow("Hand Gesture Recognition v3.0 - Press Q to Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('m'):
                mode_idx = (mode_idx + 1) % len(modes)
                print("Mode -> " + modes[mode_idx].value.upper())

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended. Goodbye!")


# ------------------------------------
#  CLI ENTRY POINT
# ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition - Python 3.13 compatible"
    )
    parser.add_argument("--multi", action="store_true",
                        help="Enable 2-hand detection")
    parser.add_argument("--app", choices=["display", "volume", "slides"],
                        default="display")
    args = parser.parse_args()

    main(
        max_hands=2 if args.multi else 1,
        app_mode=AppMode(args.app),
    )
