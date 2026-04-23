# ==============================================================================
#  Hand Gesture Recognition + Air Writing System v4.0
#  Python 3.13 + mediapipe 0.10.33  (Tasks API - no mp.solutions)
#
#  GESTURES:
#    Index only        = DRAW mode  (write in air)
#    Index + Middle    = ERASE cursor / pause drawing
#    Fist (0 fingers)  = CLEAR canvas
#    3 fingers         = CHANGE draw color (cycles)
#    4 fingers         = CHANGE brush size
#    5 fingers (open)  = SAVE canvas as PNG
#    Thumb only        = UNDO last stroke
#    Pinch (thumb+idx) = MOVE/PAN canvas
#
#  MODES  (press M to cycle):
#    DRAW    - air writing on overlay canvas
#    VOLUME  - finger count controls volume level
#    SLIDES  - finger count controls slide actions
#    DISPLAY - raw gesture display only
#
#  KEYS:
#    Q / ESC  = Quit
#    M        = Cycle mode
#    C        = Clear canvas
#    S        = Save canvas
#    Z        = Undo last stroke
#
#  INSTALL:
#    pip install mediapipe opencv-python numpy
#
#  RUN:
#    python riya_code_py313.py
#    python riya_code_py313.py --multi
# ==============================================================================

import cv2
import numpy as np
import time
import argparse
import urllib.request
import os
import math
from collections import deque, Counter
from enum import Enum

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ------------------------------------------------------------------------------
#  MODEL
# ------------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~5 MB), please wait...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
        print("")

# ------------------------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------------------------
class Cfg:
    # MediaPipe
    MIN_DETECT   = 0.72
    MIN_PRESENCE = 0.72
    MIN_TRACK    = 0.72
    SMOOTH_N     = 9          # frames for gesture majority vote

    # UI
    C_PRIMARY  = (0, 220, 180)
    C_WHITE    = (255, 255, 255)
    C_BLACK    = (0,   0,   0)
    C_DARK     = (18,  18,  28)
    C_LEFT     = (255, 140,  0)
    C_RIGHT    = (0,  200, 255)
    C_RED      = (0,   60, 220)
    C_GREEN    = (0,  200,  60)
    C_YELLOW   = (0,  220, 220)

    # Drawing palette  (BGR)
    DRAW_COLORS = [
        (0,   255, 255),   # Yellow
        (255,  80,  80),   # Blue
        (80,  255,  80),   # Green
        (80,   80, 255),   # Red
        (255, 100, 255),   # Pink
        (255, 255, 255),   # White
    ]
    BRUSH_SIZES  = [3, 6, 10, 16, 24]

    # Pinch threshold (normalised distance)
    PINCH_THRESH = 0.055
    # Min pixel movement to register a draw point
    DRAW_MIN_MOVE = 4


class AppMode(Enum):
    DRAW    = "draw"
    DISPLAY = "display"
    VOLUME  = "volume"
    SLIDES  = "slides"


# ------------------------------------------------------------------------------
#  LANDMARK INDICES
# ------------------------------------------------------------------------------
WRIST      =  0
THUMB_CMC  =  1; THUMB_MCP  =  2; THUMB_IP  =  3; THUMB_TIP  =  4
INDEX_MCP  =  5; INDEX_PIP  =  6; INDEX_DIP  =  7; INDEX_TIP  =  8
MIDDLE_MCP =  9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,17),
]


# ------------------------------------------------------------------------------
#  GESTURE BUFFER  (smoothed majority-vote)
# ------------------------------------------------------------------------------
class GestureBuffer:
    __slots__ = ("_buf",)

    def __init__(self, n=Cfg.SMOOTH_N):
        self._buf = deque(maxlen=n)

    def push(self, v):
        self._buf.append(v)

    @property
    def stable(self):
        return Counter(self._buf).most_common(1)[0][0] if self._buf else 0

    @property
    def stability(self):
        if len(self._buf) < 2:
            return 1.0
        return Counter(self._buf).most_common(1)[0][1] / len(self._buf)


# ------------------------------------------------------------------------------
#  GEOMETRY HELPERS
# ------------------------------------------------------------------------------
def lm_px(lm, w, h):
    """Return pixel coords of a single landmark."""
    return int(lm.x * w), int(lm.y * h)

def dist_norm(a, b):
    """Normalised euclidean distance between two landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ------------------------------------------------------------------------------
#  FINGER ANALYSIS
# ------------------------------------------------------------------------------
def analyse_hand(lm, side):
    """
    Returns:
        finger_states : list[bool] * 5  [thumb, index, middle, ring, pinky]
        count         : int 0-5
        confidence    : float 0-1
        pinch         : bool  (thumb tip close to index tip)
        pointing      : bool  (only index up)
        peace         : bool  (index + middle up, others down)
        fist          : bool  (0 fingers)
        thumb_only    : bool
    """
    states = []

    # Thumb: x-axis (mirrored)
    if side == "Right":
        states.append(lm[THUMB_TIP].x < lm[THUMB_IP].x)
    else:
        states.append(lm[THUMB_TIP].x > lm[THUMB_IP].x)

    # Four fingers: tip above pip in y
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        states.append(lm[tip].y < lm[pip].y)

    count = sum(states)

    # Confidence via z-depth spread of fingertips
    z = max(abs(lm[t].z) for t in TIPS[1:])
    conf = round(max(0.5, 1.0 - min(z * 3, 0.5)), 2)

    pinch      = dist_norm(lm[THUMB_TIP], lm[INDEX_TIP]) < Cfg.PINCH_THRESH
    pointing   = states[1] and not states[2] and not states[3] and not states[4]
    peace      = states[1] and states[2] and not states[3] and not states[4] and not states[0]
    fist       = count == 0
    thumb_only = states[0] and not any(states[1:])

    return states, count, conf, pinch, pointing, peace, fist, thumb_only


# ------------------------------------------------------------------------------
#  AIR-WRITING ENGINE
# ------------------------------------------------------------------------------
class AirCanvas:
    """
    Maintains a persistent drawing overlay.
    Strokes are stored so undo works.
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self._canvas   = np.zeros((h, w, 3), dtype=np.uint8)
        self._strokes  = []          # list of (points, color, thickness)
        self._cur_pts  = []
        self._cur_col  = Cfg.DRAW_COLORS[0]
        self._cur_thick= Cfg.BRUSH_SIZES[1]
        self._col_idx  = 0
        self._sz_idx   = 1
        self._prev_pt  = None
        self._drawing  = False
        self._gesture_lock_frames = 0   # prevent rapid gesture re-trigger

    # ---- public API ----

    def set_color_idx(self, idx):
        self._col_idx  = idx % len(Cfg.DRAW_COLORS)
        self._cur_col  = Cfg.DRAW_COLORS[self._col_idx]

    def next_color(self):
        self.set_color_idx(self._col_idx + 1)

    def set_size_idx(self, idx):
        self._sz_idx   = idx % len(Cfg.BRUSH_SIZES)
        self._cur_thick= Cfg.BRUSH_SIZES[self._sz_idx]

    def next_size(self):
        self.set_size_idx(self._sz_idx + 1)

    @property
    def current_color(self):
        return self._cur_col

    @property
    def current_thickness(self):
        return self._cur_thick

    @property
    def color_idx(self):
        return self._col_idx

    @property
    def size_idx(self):
        return self._sz_idx

    def start_stroke(self, pt):
        self._drawing  = True
        self._cur_pts  = [pt]
        self._prev_pt  = pt

    def add_point(self, pt):
        if not self._drawing:
            return
        if self._prev_pt is None:
            self._prev_pt = pt
            return
        dx = pt[0] - self._prev_pt[0]
        dy = pt[1] - self._prev_pt[1]
        if math.hypot(dx, dy) < Cfg.DRAW_MIN_MOVE:
            return
        cv2.line(self._canvas, self._prev_pt, pt, self._cur_col, self._cur_thick, cv2.LINE_AA)
        self._cur_pts.append(pt)
        self._prev_pt = pt

    def end_stroke(self):
        if self._drawing and len(self._cur_pts) > 1:
            self._strokes.append((list(self._cur_pts), self._cur_col, self._cur_thick))
        self._drawing  = False
        self._prev_pt  = None
        self._cur_pts  = []

    def undo(self):
        if not self._strokes:
            return
        self._strokes.pop()
        self._redraw()

    def clear(self):
        self._strokes.clear()
        self._cur_pts.clear()
        self._canvas[:] = 0
        self._prev_pt   = None
        self._drawing   = False

    def save(self, path="canvas_save.png"):
        cv2.imwrite(path, self._canvas)
        return path

    def blend(self, frame):
        """Blend canvas onto frame using mask."""
        mask = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg  = cv2.bitwise_and(frame,   frame,   mask=mask_inv)
        fg  = cv2.bitwise_and(self._canvas, self._canvas, mask=mask)
        return cv2.add(bg, fg)

    # ---- private ----
    def _redraw(self):
        self._canvas[:] = 0
        for pts, col, thick in self._strokes:
            for i in range(1, len(pts)):
                cv2.line(self._canvas, pts[i-1], pts[i], col, thick, cv2.LINE_AA)


# ------------------------------------------------------------------------------
#  DRAWING HELPERS
# ------------------------------------------------------------------------------
def draw_skeleton(frame, lm_list, color, w, h):
    pts = [lm_px(lm, w, h) for lm in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 160, 160), 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        r = 5 if i in (INDEX_TIP, MIDDLE_TIP, THUMB_TIP) else 3
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, Cfg.C_WHITE, 1, cv2.LINE_AA)


def draw_corner_box(frame, x1, y1, x2, y2, color, t=2, cl=18):
    for (px, py), (dx, dy) in [
        ((x1,y1),(cl,0)),((x1,y1),(0,cl)),
        ((x2,y1),(-cl,0)),((x2,y1),(0,cl)),
        ((x1,y2),(cl,0)),((x1,y2),(0,-cl)),
        ((x2,y2),(-cl,0)),((x2,y2),(0,-cl)),
    ]:
        cv2.line(frame, (px,py), (px+dx, py+dy), color, t, cv2.LINE_AA)


def put_text_shadow(frame, text, pos, scale, color, thick=1):
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, Cfg.C_BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_palette(frame, canvas, w):
    """Draw color/size palette strip at top-right."""
    px = w - 30
    for i, col in enumerate(Cfg.DRAW_COLORS):
        py = 15 + i * 28
        cv2.circle(frame, (px, py), 10, col, -1, cv2.LINE_AA)
        if i == canvas.color_idx:
            cv2.circle(frame, (px, py), 13, Cfg.C_WHITE, 2, cv2.LINE_AA)

    # Brush size indicator
    py2 = 15 + len(Cfg.DRAW_COLORS) * 28 + 10
    cv2.circle(frame, (px, py2), canvas.current_thickness, canvas.current_color, -1, cv2.LINE_AA)
    put_text_shadow(frame, "B" + str(canvas.size_idx+1), (px-8, py2+4), 0.3, Cfg.C_WHITE)


def draw_hud(frame, fps, mode, w, h, gesture_name, canvas=None):
    """Top-left info panel + bottom bar."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 80), Cfg.C_DARK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    put_text_shadow(frame, "FPS: " + str(round(fps, 1)), (10, 24), 0.62, Cfg.C_PRIMARY)
    put_text_shadow(frame, "MODE: " + mode.value.upper(),  (10, 50), 0.55, Cfg.C_WHITE)
    put_text_shadow(frame, gesture_name,                   (10, 74), 0.48, Cfg.C_YELLOW)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h-32), (w, h), Cfg.C_DARK, -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame,
        "Q:Quit  M:Mode  C:Clear  S:Save  Z:Undo  |  "
        "1-finger=Draw  2=Pause  Fist=Clear  3=Color  4=Size  5=Save  Thumb=Undo",
        (6, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140,140,160), 1, cv2.LINE_AA)

    if canvas and mode == AppMode.DRAW:
        draw_palette(frame, canvas, w)


def draw_hand_info(frame, x_min, x_max, y_min, y_max,
                   label, hand_color, stable_count,
                   finger_states, confidence, stability, mode):
    h_frame, w_frame = frame.shape[:2]

    # Big count
    ct = str(stable_count)
    ts = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 6)[0]
    tx = x_min + (x_max - x_min)//2 - ts[0]//2
    ty = max(y_min - 12, 75)
    cv2.putText(frame, ct, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, 3.2, Cfg.C_BLACK, 8, cv2.LINE_AA)
    cv2.putText(frame, ct, (tx,   ty),   cv2.FONT_HERSHEY_SIMPLEX, 3.2, hand_color,  6, cv2.LINE_AA)

    cv2.putText(frame, label + " Hand", (x_min, y_max+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 1, cv2.LINE_AA)

    # Confidence bar
    bx, by = x_min, y_max + 32
    bw = x_max - x_min
    cv2.rectangle(frame, (bx, by), (bx+bw, by+7), (50,50,70), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(bw*confidence), by+7),
                  lerp_color((0,80,220),(0,220,80), confidence), -1)
    cv2.putText(frame, "Conf " + str(int(confidence*100)) + "%",
                (bx, by+22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Cfg.C_WHITE, 1, cv2.LINE_AA)

    # Stability dot
    sc = (0,220,80) if stability>=0.85 else ((0,200,220) if stability>=0.6 else (0,80,220))
    sl = "Stable" if stability>=0.85 else ("Settling" if stability>=0.6 else "Jitter")
    cv2.putText(frame, "* " + sl, (bx, by+38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sc, 1, cv2.LINE_AA)

    # Finger dots
    dy   = y_max + 58
    gap  = 17
    x0   = x_min + (bw - gap*4)//2
    for fi, (st, lb) in enumerate(zip(finger_states, ["T","I","M","R","P"])):
        dx = x0 + fi * gap
        cv2.circle(frame, (dx, dy), 6, (0,220,100) if st else (70,70,110), -1)
        cv2.putText(frame, lb, (dx-4, dy+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, Cfg.C_WHITE, 1, cv2.LINE_AA)


# ------------------------------------------------------------------------------
#  MODE-SPECIFIC OVERLAYS
# ------------------------------------------------------------------------------
VOLUME_LABELS = {0:"MUTE",1:"Vol 20%",2:"Vol 40%",3:"Vol 60%",4:"Vol 80%",5:"Vol 100%"}
SLIDES_LABELS = {0:"PAUSE",1:"PREV",2:"NEXT",3:"ZOOM IN",4:"BLANK",5:"PRESENT"}

def draw_mode_banner(frame, count, mode, w, h):
    if mode == AppMode.VOLUME:
        label = VOLUME_LABELS.get(count, "")
        # Visual volume bar
        vol = count / 5.0
        bw  = 200
        bx  = w//2 - bw//2
        by  = h - 75
        cv2.rectangle(frame, (bx-2, by-2), (bx+bw+2, by+18), (40,40,60), -1)
        cv2.rectangle(frame, (bx, by), (bx+int(bw*vol), by+14),
                      lerp_color((0,80,200),(0,220,80), vol), -1)
        put_text_shadow(frame, label, (bx, by-8), 0.6, Cfg.C_PRIMARY, 2)

    elif mode == AppMode.SLIDES:
        label = SLIDES_LABELS.get(count, "")
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2-120, h-90), (w//2+120, h-55), (30,30,50), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        put_text_shadow(frame, ">> " + label, (w//2-100, h-64), 0.75, Cfg.C_PRIMARY, 2)


# ------------------------------------------------------------------------------
#  DRAW MODE GESTURE HANDLER
# ------------------------------------------------------------------------------
class DrawController:
    """Translates hand gesture data into canvas operations."""

    COOLDOWN = 18   # frames between gesture-triggered actions

    def __init__(self, canvas):
        self.canvas   = canvas
        self._cd      = 0          # cooldown counter
        self._was_drawing = False
        self._status  = ""         # message shown on screen
        self._status_t= 0

    def status(self):
        if time.time() - self._status_t < 1.8:
            return self._status
        return ""

    def _set_status(self, msg):
        self._status   = msg
        self._status_t = time.time()

    def update(self, lm, w, h, finger_states, count,
               pinch, pointing, peace, fist, thumb_only):
        """
        Call once per frame per hand.
        Returns cursor_pt (or None), is_drawing (bool).
        """
        if self._cd > 0:
            self._cd -= 1

        tip_px = lm_px(lm[INDEX_TIP], w, h)
        cursor = tip_px

        # --- Gesture logic ---

        # FIST -> clear
        if fist and self._cd == 0:
            if self._was_drawing:
                self.canvas.end_stroke()
                self._was_drawing = False
            self.canvas.clear()
            self._cd = self.COOLDOWN
            self._set_status("Canvas cleared")
            return cursor, False

        # THUMB ONLY -> undo
        if thumb_only and self._cd == 0:
            if self._was_drawing:
                self.canvas.end_stroke()
                self._was_drawing = False
            self.canvas.undo()
            self._cd = self.COOLDOWN
            self._set_status("Undo")
            return cursor, False

        # 3 fingers -> next color
        if count == 3 and self._cd == 0:
            if self._was_drawing:
                self.canvas.end_stroke()
                self._was_drawing = False
            self.canvas.next_color()
            self._cd = self.COOLDOWN
            self._set_status("Color changed")
            return cursor, False

        # 4 fingers -> next brush size
        if count == 4 and self._cd == 0:
            if self._was_drawing:
                self.canvas.end_stroke()
                self._was_drawing = False
            self.canvas.next_size()
            self._cd = self.COOLDOWN
            self._set_status("Brush size: " + str(self.canvas.current_thickness))
            return cursor, False

        # 5 fingers -> save
        if count == 5 and self._cd == 0:
            if self._was_drawing:
                self.canvas.end_stroke()
                self._was_drawing = False
            path = self.canvas.save()
            self._cd = self.COOLDOWN * 3
            self._set_status("Saved: " + path)
            return cursor, False

        # PEACE (2 fingers) -> pause drawing
        if peace:
            if self._was_drawing:
                self.canvas.end_stroke()
                self._was_drawing = False
            return cursor, False

        # INDEX ONLY -> draw
        if pointing:
            if not self._was_drawing:
                self.canvas.start_stroke(tip_px)
                self._was_drawing = True
            else:
                self.canvas.add_point(tip_px)
            return cursor, True

        # Anything else -> stop drawing
        if self._was_drawing:
            self.canvas.end_stroke()
            self._was_drawing = False

        return cursor, False


# ------------------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------------------
def main(max_hands=2):
    print("==========================================")
    print("  Hand Gesture + Air Writing System v4.0 ")
    print("  Python 3.13 Compatible                 ")
    print("  Q:Quit  M:Mode  C:Clear  S:Save  Z:Undo")
    print("==========================================")
    print("")

    ensure_model()

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=max_hands,
        min_hand_detection_confidence=Cfg.MIN_DETECT,
        min_hand_presence_confidence=Cfg.MIN_PRESENCE,
        min_tracking_confidence=Cfg.MIN_TRACK,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    canvas      = AirCanvas(W, H)
    draw_ctrl   = DrawController(canvas)
    g_bufs      = {}                       # hand_idx -> GestureBuffer
    modes       = list(AppMode)
    mode_idx    = 0
    fps_time    = time.time()
    fps         = 0.0
    frame_cnt   = 0
    gesture_name= ""

    with mp_vision.HandLandmarker.create_from_options(options) as lander:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # -- Detection --
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int(time.time() * 1000)
            result   = lander.detect_for_video(mp_img, ts_ms)

            # -- FPS --
            frame_cnt += 1
            now = time.time()
            if now - fps_time >= 0.5:
                fps       = frame_cnt / (now - fps_time)
                frame_cnt = 0
                fps_time  = now

            mode = modes[mode_idx]

            # -- Per-hand processing --
            lm_lists    = result.hand_landmarks
            hand_labels = result.handedness
            cursor_pt   = None
            is_drawing  = False

            for idx, (lm_list, hness) in enumerate(zip(lm_lists, hand_labels)):
                label      = hness[0].category_name
                hand_color = Cfg.C_RIGHT if label == "Right" else Cfg.C_LEFT

                states, count, conf, pinch, pointing, peace, fist, thumb_only \
                    = analyse_hand(lm_list, label)

                if idx not in g_bufs:
                    g_bufs[idx] = GestureBuffer()
                g_bufs[idx].push(count)
                stable = g_bufs[idx].stable
                stab   = g_bufs[idx].stability

                # Skeleton + box
                draw_skeleton(frame, lm_list, hand_color, w, h)
                xs    = [int(lm.x * w) for lm in lm_list]
                ys    = [int(lm.y * h) for lm in lm_list]
                x1,x2 = max(0,min(xs)-14), min(w,max(xs)+14)
                y1,y2 = max(0,min(ys)-14), min(h,max(ys)+14)
                draw_corner_box(frame, x1, x2, y1, y2, hand_color)
                draw_hand_info(frame, x1, x2, y1, y2,
                               label, hand_color, stable,
                               states, conf, stab, mode)

                # Gesture name for HUD
                if pointing:   gesture_name = "DRAW"
                elif peace:    gesture_name = "PAUSE"
                elif fist:     gesture_name = "FIST - CLEAR"
                elif thumb_only: gesture_name = "THUMB - UNDO"
                elif pinch:    gesture_name = "PINCH"
                elif count==3: gesture_name = "3 - CHANGE COLOR"
                elif count==4: gesture_name = "4 - BRUSH SIZE"
                elif count==5: gesture_name = "5 - SAVE"
                else:          gesture_name = str(stable) + " fingers"

                # Mode actions
                if mode == AppMode.DRAW:
                    cp, isd = draw_ctrl.update(
                        lm_list, w, h, states, stable,
                        pinch, pointing, peace, fist, thumb_only)
                    if cp:
                        cursor_pt  = cp
                        is_drawing = isd
                else:
                    draw_mode_banner(frame, stable, mode, w, h)

            # -- Blend canvas onto frame --
            if mode == AppMode.DRAW:
                frame = canvas.blend(frame)

                # Draw cursor
                if cursor_pt:
                    col = canvas.current_color if is_drawing else Cfg.C_WHITE
                    r   = canvas.current_thickness + 4
                    cv2.circle(frame, cursor_pt, r, col, 2, cv2.LINE_AA)
                    cv2.circle(frame, cursor_pt, 3, col, -1, cv2.LINE_AA)

                # Status message
                status = draw_ctrl.status()
                if status:
                    put_text_shadow(frame, status,
                                    (w//2 - 120, h//2 - 20), 0.8, Cfg.C_YELLOW, 2)

            if not lm_lists:
                gesture_name = "No hand detected"

            # -- HUD --
            draw_hud(frame, fps, mode, w, h, gesture_name,
                     canvas if mode == AppMode.DRAW else None)

            cv2.imshow("Hand Gesture + Air Writing v4.0 - Q to Quit", frame)

            # -- Keys --
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('m'):
                mode_idx = (mode_idx + 1) % len(modes)
                print("Mode -> " + modes[mode_idx].value.upper())
            elif key == ord('c'):
                canvas.clear()
                print("Canvas cleared.")
            elif key == ord('s'):
                p = canvas.save()
                print("Saved: " + p)
            elif key == ord('z'):
                canvas.undo()
                print("Undo.")

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


# ------------------------------------------------------------------------------
#  ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hand Gesture + Air Writing - Python 3.13"
    )
    parser.add_argument("--multi", action="store_true",
                        help="Enable 2-hand tracking (default: 1)")
    args = parser.parse_args()
    main(max_hands=2 if args.multi else 2)