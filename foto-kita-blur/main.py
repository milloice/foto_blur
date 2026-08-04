import os
import sys
import time
import math
import glob
import random
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR = os.path.join(BASE_DIR, "emojis")
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

os.makedirs(EMOJI_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)

# ----------------------------------------------------
# MediaPipe HandLandmarker Auto-Downloader
# ----------------------------------------------------
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe HandLandmarker model (hand_landmarker.task)...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")

ensure_model_exists()

# Try importing MediaPipe Tasks
MP_TASKS_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_TASKS_AVAILABLE = True
except Exception as e:
    print(f"MediaPipe Tasks import warning: {e}")


# 21 Hand Landmarks Skeleton Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm bridge
]


# ----------------------------------------------------
# Neural Network Hand Tracker (Google MediaPipe 21 Landmarks)
# ----------------------------------------------------
class HandTracker:
    def __init__(self):
        self.gesture_history = []
        self.history_size = 5
        self.detector = None

        if MP_TASKS_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.6,
                    min_hand_presence_confidence=0.6,
                    min_tracking_confidence=0.6
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                print("HandTracker initialized with Google MediaPipe Neural Network (21 3D Landmarks).")
            except Exception as e:
                print(f"HandLandmarker init error: {e}")

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        default_cx, default_cy = w // 2, h // 2
        v_detected = False
        hand_x, hand_y = default_cx, default_cy
        all_hand_landmarks = []

        if self.detector is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self.detector.detect(mp_image)

            if res and res.hand_landmarks:
                for hand_lms in res.hand_landmarks:
                    all_hand_landmarks.append(hand_lms)
                    if self.check_v_gesture_mp(hand_lms):
                        v_detected = True
                        hand_x = int(((hand_lms[8].x + hand_lms[12].x) / 2) * w)
                        hand_y = int(min(hand_lms[8].y, hand_lms[12].y) * h)
                        break
                if not v_detected and res.hand_landmarks:
                    hand_lms = res.hand_landmarks[0]
                    hand_x = int(hand_lms[0].x * w)
                    hand_y = int(hand_lms[0].y * h)

        # Smooth detection history
        self.gesture_history.append(v_detected)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)

        is_v_stable = sum(self.gesture_history) >= (self.history_size // 2)
        return is_v_stable, hand_x, hand_y, all_hand_landmarks

    def check_v_gesture_mp(self, lms):
        index_up = lms[8].y < lms[6].y and lms[8].y < lms[5].y
        middle_up = lms[12].y < lms[10].y and lms[12].y < lms[9].y

        ring_down = lms[16].y > lms[14].y
        pinky_down = lms[20].y > lms[18].y

        dx = lms[8].x - lms[12].x
        dy = lms[8].y - lms[12].y
        tip_dist = math.hypot(dx, dy)

        return index_up and middle_up and ring_down and pinky_down and tip_dist > 0.035

    def draw_hand_landmarks(self, frame_bgr, all_hand_landmarks):
        if not all_hand_landmarks:
            return

        h, w = frame_bgr.shape[:2]

        for hand_lms in all_hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

            # 1. Draw Skeleton Lines
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pts) and p2 < len(pts):
                    cv2.line(frame_bgr, pts[p1], pts[p2], (255, 255, 0), 2, cv2.LINE_AA)

            # 2. Draw 21 Tracking Dots
            for i, (px, py) in enumerate(pts):
                if i in [4, 8, 12, 16, 20]:  # Finger Tips
                    cv2.circle(frame_bgr, (px, py), 9, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, (px, py), 12, (255, 255, 255), 2, cv2.LINE_AA)
                elif i == 0:  # Wrist
                    cv2.circle(frame_bgr, (px, py), 10, (0, 215, 255), -1, cv2.LINE_AA)
                else:  # Joints
                    cv2.circle(frame_bgr, (px, py), 6, (255, 105, 180), -1, cv2.LINE_AA)


# ----------------------------------------------------
# Emoji & User Photo Floating Manager (Enlarged & Moving)
# ----------------------------------------------------
class FloatingAssetManager:
    def __init__(self, emoji_dir, photos_dir, max_items=40):
        self.emoji_dir = emoji_dir
        self.photos_dir = photos_dir
        self.max_items = max_items
        self.asset_pool = []
        self.active_items = []
        self.load_all_assets()

    def load_all_assets(self):
        # 1. Load Emoji PNGs
        png_files = glob.glob(os.path.join(self.emoji_dir, "*.png"))
        for p in png_files:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                self.asset_pool.append(img)

        # 2. Load & Format User Photos into circular framed PNG assets
        valid_exts = ("*.png", "*.jpg", "*.jpeg")
        photo_files = []
        for ext in valid_exts:
            photo_files.extend(glob.glob(os.path.join(self.photos_dir, ext)))

        for pf in photo_files:
            p_img = cv2.imread(pf, cv2.IMREAD_UNCHANGED)
            if p_img is not None:
                formatted_photo = self.format_user_photo(p_img)
                if formatted_photo is not None:
                    # Add multiple copies of user photo so it shows up frequently
                    for _ in range(3):
                        self.asset_pool.append(formatted_photo)

        print(f"Loaded {len(self.asset_pool)} total floating assets (emojis + photos).")

    def format_user_photo(self, photo_bgr):
        """Converts user photo into a cute circular framed image with pink/gold border & alpha channel."""
        h, w = photo_bgr.shape[:2]
        crop_sz = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = photo_bgr[max(0, cy - crop_sz//2) : cy + crop_sz//2, max(0, cx - crop_sz//2) : cx + crop_sz//2]

        sz = 200
        resized = cv2.resize(cropped, (sz, sz), interpolation=cv2.INTER_AREA)

        # Create PIL circular crop with border
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        mask = Image.new("L", (sz, sz), 0)
        draw_m = ImageDraw.Draw(mask)
        draw_m.ellipse([10, 10, sz-10, sz-10], fill=255)

        output = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        output.paste(pil_img, (0, 0), mask=mask)

        # Draw glowing pink border
        draw_out = ImageDraw.Draw(output)
        draw_out.ellipse([5, 5, sz-5, sz-5], outline=(255, 105, 180, 255), width=8)
        draw_out.ellipse([1, 1, sz-1, sz-1], outline=(255, 215, 0, 255), width=3)

        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2BGRA)
        return res_np

    def spawn_item(self, hand_x, hand_y, screen_w, screen_h):
        if not self.asset_pool or len(self.active_items) >= self.max_items:
            return

        img = random.choice(self.asset_pool)
        # ENLARGED SIZE RANGE: 90px to 160px
        size = random.randint(90, 160)
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        if random.random() < 0.7 and hand_x > 0 and hand_y > 0:
            x = hand_x + random.randint(-160, 160)
            y = hand_y + random.randint(-40, 80)
        else:
            x = random.randint(20, max(21, screen_w - size - 20))
            y = screen_h + random.randint(0, 30)

        x = max(10, min(screen_w - size - 10, x))

        self.active_items.append({
            "image": resized,
            "x": float(x),
            "base_x": float(x),
            "y": float(y),
            "size": size,
            "speed_y": random.uniform(7.0, 15.0),
            "amplitude_x": random.uniform(20, 50),
            "frequency_x": random.uniform(0.03, 0.08),
            "phase_x": random.uniform(0, 2 * math.pi)
        })

    def update_and_render(self, frame_bgr, v_gesture_active, hand_x, hand_y):
        if not self.asset_pool:
            return frame_bgr

        # Clear ALL floating items instantly when V gesture is not active
        if not v_gesture_active:
            self.active_items = []
            return frame_bgr

        screen_h, screen_w = frame_bgr.shape[0], frame_bgr.shape[1]

        # Spawn new items instantly and rapidly when V gesture is active
        for _ in range(random.randint(3, 6)):
            self.spawn_item(hand_x, hand_y, screen_w, screen_h)

        remaining_items = []
        for item in self.active_items:
            item["y"] -= item["speed_y"]
            item["phase_x"] += item["frequency_x"]
            item["x"] = item["base_x"] + math.sin(item["phase_x"]) * item["amplitude_x"]

            if item["y"] > -item["size"] - 20:
                self.overlay_png(frame_bgr, item["image"], int(item["x"]), int(item["y"]))
                remaining_items.append(item)

        self.active_items = remaining_items
        return frame_bgr

    @staticmethod
    def overlay_png(bg, fg, x, y):
        h, w = fg.shape[:2]
        bg_h, bg_w = bg.shape[:2]

        if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
            return

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

        fg_x1, fg_y1 = x1 - x, y1 - y
        fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)

        alpha = fg[fg_y1:fg_y2, fg_x1:fg_x2, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        fg_bgr = fg[fg_y1:fg_y2, fg_x1:fg_x2, :3]
        bg_crop = bg[y1:y2, x1:x2]

        bg[y1:y2, x1:x2] = (fg_bgr * alpha + bg_crop * (1.0 - alpha)).astype(np.uint8)


# ----------------------------------------------------
# Glitter & Sparkle Effect
# ----------------------------------------------------
class GlitterEffect:
    def __init__(self, num_particles=70):
        self.num_particles = num_particles
        self.particles = []

    def init_particles(self, screen_w, screen_h):
        self.particles = []
        colors = [
            (255, 255, 255),    # White
            (203, 192, 255),    # Pinkish
            (200, 240, 255),    # Light gold/yellow
            (255, 220, 240)     # Pastel violet
        ]
        for _ in range(self.num_particles):
            self.particles.append({
                "x": random.randint(0, screen_w),
                "y": random.randint(0, screen_h),
                "size": random.randint(2, 6),
                "color": random.choice(colors),
                "phase": random.uniform(0, 2 * math.pi),
                "speed": random.uniform(0.08, 0.18),
                "vy": random.uniform(-0.8, -0.2)
            })

    def render(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if not self.particles:
            self.init_particles(w, h)

        overlay = frame_bgr.copy()

        for p in self.particles:
            p["phase"] += p["speed"]
            p["y"] += p["vy"]
            if p["y"] < 0:
                p["y"] = h
                p["x"] = random.randint(0, w)

            color_bgr = p["color"]
            cx, cy = int(p["x"]), int(p["y"])
            sz = int(p["size"])

            cv2.circle(overlay, (cx, cy), sz, color_bgr, -1)
            cv2.line(overlay, (cx - sz * 2, cy), (cx + sz * 2, cy), color_bgr, 1)
            cv2.line(overlay, (cx, cy - sz * 2), (cx, cy + sz * 2), color_bgr, 1)

        cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)


# ----------------------------------------------------
# Simplified & Sleek Top UI Header
# ----------------------------------------------------
def draw_ui_banner(frame_bgr, is_v_active):
    h, w = frame_bgr.shape[:2]

    # Minimal floating top-center pill badge
    badge_w, badge_h = 320, 46
    bx1 = (w - badge_w) // 2
    by1 = 20
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    # Glassmorphic rounded rectangle badge
    badge_crop = frame_bgr[by1:by2, bx1:bx2].copy()
    overlay_col = (40, 10, 70) if is_v_active else (20, 20, 20)
    cv2.rectangle(badge_crop, (0, 0), (badge_w, badge_h), overlay_col, -1)
    cv2.addWeighted(badge_crop, 0.75, frame_bgr[by1:by2, bx1:bx2], 0.25, 0, frame_bgr[by1:by2, bx1:bx2])

    border_col = (255, 105, 180) if is_v_active else (150, 150, 150)
    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), border_col, 2)

    # Render simple text via PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"
    try:
        font = ImageFont.truetype(font_path, 22)
    except:
        font = ImageFont.load_default()

    if is_v_active:
        text = "✌️ V GESTURE ACTIVE 💕"
        text_color = (255, 105, 180)
    else:
        text = "✌️ Show V Gesture"
        text_color = (255, 255, 255)

    # Center text in badge
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (badge_w - tw) // 2
    ty = (badge_h - th) // 2

    draw.text((tx, ty), text, fill=text_color, font=font)
    frame_bgr[by1:by2, bx1:bx2] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------
# Main Program Loop
# ----------------------------------------------------
def main():
    print("Initializing Hand Tracking V-Gesture Program...")

    cap = None
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            break

    if cap is None or not cap.isOpened():
        print("ERROR: Could not open webcam. Please connect a webcam and try again.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    assets_mgr = FloatingAssetManager(EMOJI_DIR, PHOTOS_DIR, max_items=45)
    glitter = GlitterEffect(num_particles=65)

    win_name = "Fotoblur - V Gesture Love Effect"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    blur_factor = 0.0
    smooth_hand_x = 640.0
    smooth_hand_y = 360.0

    print("Program Ready! Show V Gesture (peace sign) to trigger effect.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process hand tracking & gesture detection using 21 MediaPipe Neural Network Landmarks
        is_v_gesture, hand_x, hand_y, all_landmarks = tracker.process(frame)

        # Smooth hand position tracking
        smooth_hand_x += (hand_x - smooth_hand_x) * 0.3
        smooth_hand_y += (hand_y - smooth_hand_y) * 0.3

        # Smooth blur factor transition (0.0 = clear, 1.0 = full blur)
        target_blur = 1.0 if is_v_gesture else 0.0
        blur_factor += (target_blur - blur_factor) * 0.25

        # ----------------------------------------------------
        # 1. Full Camera Gaussian Blur when V-Gesture Active
        # ----------------------------------------------------
        if blur_factor > 0.01:
            k_size = int(51 * blur_factor) | 1
            blurred_frame = cv2.GaussianBlur(frame, (k_size, k_size), 0)
            frame = cv2.addWeighted(blurred_frame, blur_factor, frame, 1.0 - blur_factor, 0)

            # Render love emojis & user photos floating up (spawn ONLY during V-gesture)
            frame = assets_mgr.update_and_render(frame, is_v_gesture, int(smooth_hand_x), int(smooth_hand_y))

            # Render glitter sparkles
            glitter.render(frame)

        # ----------------------------------------------------
        # 2. Draw 21 Neural Network Hand Tracking Dots & Skeleton
        # ----------------------------------------------------
        tracker.draw_hand_landmarks(frame, all_landmarks)

        # 3. Draw simplified top UI badge
        draw_ui_banner(frame, is_v_gesture)

        # Show frame fullscreen
        cv2.imshow(win_name, frame)

        # Exit on 'Q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")


if __name__ == "__main__":
    main()
