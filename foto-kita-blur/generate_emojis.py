import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOJI_DIR = os.path.join(os.path.dirname(__file__), "emojis")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "photos")

os.makedirs(EMOJI_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)

def create_heart_png(filename, fill_color, border_color=None, sparkle=False, inner_text=None, size=(128, 128)):
    """Generates a high-quality smooth PNG heart image with alpha transparency."""
    w, h = size
    # Create 4x anti-aliased image
    scale = 4
    sw, sh = w * scale, h * scale
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Parametric heart shape
    points = []
    num_points = 200
    cx, cy = sw / 2, sh / 2 - 10 * scale
    radius = 14 * scale

    for i in range(num_points):
        t = i * (2 * math.pi / num_points)
        # Heart formula
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        px = cx + x * radius / 16
        py = cy + y * radius / 16
        points.append((px, py))

    # Outer glow / shadow
    if border_color:
        border_points = []
        for px, py in points:
            dx, dy = px - cx, py - cy
            dist = math.hypot(dx, dy)
            border_points.append((cx + dx * 1.1, cy + dy * 1.1))
        draw.polygon(border_points, fill=border_color)

    draw.polygon(points, fill=fill_color)

    # Add shiny highlight on top left lobe
    highlight_points = []
    for i in range(130, 170):
        px, py = points[i]
        # inset
        hx = cx + (px - cx) * 0.75
        hy = cy + (py - cy) * 0.75
        highlight_points.append((hx, hy))
    if len(highlight_points) > 2:
        draw.polygon(highlight_points, fill=(255, 255, 255, 180))

    # Optional sparkles
    if sparkle:
        sparkle_centers = [
            (sw * 0.75, sh * 0.25),
            (sw * 0.2, sh * 0.35),
            (sw * 0.7, sh * 0.7),
            (sw * 0.3, sh * 0.75)
        ]
        for sx, sy in sparkle_centers:
            sp_len = 12 * scale
            draw.line([(sx - sp_len, sy), (sx + sp_len, sy)], fill=(255, 255, 255, 230), width=3*scale)
            draw.line([(sx, sy - sp_len), (sx, sy + sp_len)], fill=(255, 255, 255, 230), width=3*scale)

    # Downsample for smooth anti-aliasing
    img = img.resize(size, Image.Resampling.LANCZOS)
    out_path = os.path.join(EMOJI_DIR, filename)
    img.save(out_path, "PNG")
    print(f"Saved: {out_path}")

def render_unicode_emoji(filename, text_char, font_path="C:\\Windows\\Fonts\\seguiemj.ttf", size=(128, 128)):
    """Attempts to render Unicode emoji using Segoe UI Emoji font."""
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    try:
        font = ImageFont.truetype(font_path, int(w * 0.75))
        draw = ImageDraw.Draw(img)
        # Use textbbox to center
        bbox = draw.textbbox((0, 0), text_char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (w - tw) / 2 - bbox[0]
        ty = (h - th) / 2 - bbox[1]
        draw.text((tx, ty), text_char, font=font, embedded_color=True)
        out_path = os.path.join(EMOJI_DIR, filename)
        img.save(out_path, "PNG")
        print(f"Saved Unicode Emoji: {out_path}")
        return True
    except Exception as e:
        print(f"Unicode font render notice for {text_char}: {e}")
        return False

def generate_all_emojis():
    # Try rendering Segoe UI Emoji unicode icons first
    unicode_emojis = [
        ("emoji_heart_red.png", "❤️"),
        ("emoji_heart_pink.png", "💖"),
        ("emoji_heart_sparkle.png", "✨"),
        ("emoji_heart_two.png", "💕"),
        ("emoji_heart_revolving.png", "💞"),
        ("emoji_kiss.png", "💋"),
        ("emoji_love_face.png", "🥰")
    ]
    
    for fname, char in unicode_emojis:
        render_unicode_emoji(fname, char)

    # Always generate custom ultra-clean high-res transparent PNG hearts as well
    create_heart_png("heart_classic_red.png", fill_color=(255, 40, 90, 255), border_color=(200, 10, 50, 255), sparkle=True)
    create_heart_png("heart_hot_pink.png", fill_color=(255, 105, 180, 255), border_color=(220, 20, 140, 255), sparkle=True)
    create_heart_png("heart_soft_pink.png", fill_color=(255, 182, 193, 255), border_color=(255, 105, 180, 255), sparkle=False)
    create_heart_png("heart_purple.png", fill_color=(186, 85, 211, 255), border_color=(138, 43, 226, 255), sparkle=True)
    create_heart_png("heart_coral.png", fill_color=(255, 127, 80, 255), border_color=(233, 84, 32, 255), sparkle=True)
    create_heart_png("heart_gold.png", fill_color=(255, 215, 0, 255), border_color=(218, 165, 32, 255), sparkle=True)

    # Create a placeholder sample user photo if photo folder is empty
    sample_photo_path = os.path.join(PHOTOS_DIR, "sample_user.png")
    if not os.path.exists(sample_photo_path) and len(os.listdir(PHOTOS_DIR)) == 0:
        p_img = Image.new("RGBA", (300, 300), (255, 240, 245, 255))
        p_draw = ImageDraw.Draw(p_img)
        p_draw.ellipse([20, 20, 280, 280], fill=(255, 182, 193, 255), outline=(255, 20, 147, 255), width=6)
        try:
            f = ImageFont.truetype("arial.ttf", 36)
            p_draw.text((80, 130), "Your Photo", fill=(199, 21, 133, 255), font=f)
        except:
            p_draw.text((80, 130), "Your Photo", fill=(199, 21, 133, 255))
        p_img.save(sample_photo_path)
        print(f"Created sample photo: {sample_photo_path}")

if __name__ == "__main__":
    generate_all_emojis()
