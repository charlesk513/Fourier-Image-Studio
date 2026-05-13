"""
╔══════════════════════════════════════════════════════════════════╗
║          FOURIER IMAGE STUDIO — Upgraded Edition                 ║
║  FFT Sharpening · Smoothing · 10 Live Colour Filters             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys, threading

try:
    from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    os.system(f"{sys.executable} -m pip install Pillow --quiet")
    from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps

try:
    import numpy as np
except ImportError:
    os.system(f"{sys.executable} -m pip install numpy --quiet")
    import numpy as np


# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────

P = {
    "bg":      "#09090f",
    "surface": "#111118",
    "card":    "#17171f",
    "border":  "#27273a",
    "accent":  "#7c6eff",
    "accent2": "#00e5b0",
    "warn":    "#ff6b6b",
    "text":    "#dcdcf0",
    "muted":   "#5c5c7a",
    "white":   "#ffffff",
}

FONT_SUB  = ("Segoe UI", 8)
FONT_BTN  = ("Segoe UI", 10, "bold")
FONT_LABEL = ("Segoe UI", 9)
FONT_MONO  = ("Courier New", 8)


# ─────────────────────────────────────────────────────────────────
# FFT ENGINE  O(NM log NM)
# ─────────────────────────────────────────────────────────────────

class FourierEngine:
    @staticmethod
    def _apply_filter(channel, mask):
        f  = np.fft.fft2(channel.astype(np.float64))
        fs = np.fft.fftshift(f)
        fb = np.fft.ifftshift(fs * mask)
        return np.abs(np.fft.ifft2(fb))

    @staticmethod
    def _gauss_lp(shape, cutoff):
        rows, cols = shape
        y = np.arange(rows) - rows // 2
        x = np.arange(cols) - cols // 2
        X, Y = np.meshgrid(x, y)
        sigma = cutoff * min(rows, cols) / 2
        return np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    @staticmethod
    def _gauss_hp(shape, cutoff):
        return 1.0 - FourierEngine._gauss_lp(shape, cutoff)

    @staticmethod
    def _unsharp(shape, strength, radius):
        return 1.0 + strength * FourierEngine._gauss_hp(shape, radius)

    @staticmethod
    def _wiener(shape, strength):
        rows, cols = shape
        y = np.arange(rows) - rows // 2
        x = np.arange(cols) - cols // 2
        X, Y = np.meshgrid(x, y)
        D  = np.sqrt(X**2 + Y**2)
        md = np.sqrt((rows//2)**2 + (cols//2)**2)
        nd = D / (md + 1e-10)
        K  = 0.02 / (strength + 1e-5)
        H  = np.exp(-nd**2 * (1 - strength) * 4)
        w  = H / (H**2 + K)
        return w / (w.max() + 1e-10)

    @staticmethod
    def process(pil_img, mode, strength):
        arr   = np.array(pil_img.convert("RGB"), dtype=np.float64)
        shape = arr.shape[:2]
        res   = np.zeros_like(arr)
        if mode == "smooth":
            mask = FourierEngine._gauss_lp(shape, 0.5 - strength * 0.38)
        else:
            um   = FourierEngine._unsharp(shape, strength * 3.5,
                                           0.08 + (1 - strength) * 0.12)
            wi   = FourierEngine._wiener(shape, strength)
            mask = 0.6 * um + 0.4 * wi * (1 + strength * 2)
        for c in range(3):
            res[:, :, c] = FourierEngine._apply_filter(arr[:, :, c], mask)
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────
# COLOUR FILTER ENGINE  — 11 filters (index 0 = Original)
# ─────────────────────────────────────────────────────────────────

class FilterEngine:
    FILTERS = [
        ("Original",    "No filter applied",           "○"),
        ("Vivid",       "Punchy colours & contrast",   "◉"),
        ("Sepia",       "Warm vintage brown tone",      "◈"),
        ("Cool Mist",   "Icy blue-tinted cooldown",    "❄"),
        ("Golden Hour", "Warm sunset amber wash",      "☀"),
        ("Noir",        "High-contrast black & white", "◆"),
        ("Faded",       "Dreamy low-contrast matte",   "◎"),
        ("Cinematic",   "Film-grade teal & orange",    "▣"),
        ("Neon Pop",    "Saturated candy colours",     "✦"),
        ("Soft Bloom",  "Glowing pastel luminance",    "✿"),
        ("Deep Forest", "Rich greens & shadowed dark", "⬡"),
    ]

    @staticmethod
    def apply(pil_img, index):
        img = pil_img.convert("RGB")
        arr = np.array(img, dtype=np.float64)

        if index == 0:   # Original
            return img

        if index == 1:   # Vivid
            img = ImageEnhance.Color(img).enhance(1.8)
            img = ImageEnhance.Contrast(img).enhance(1.3)
            return ImageEnhance.Sharpness(img).enhance(1.5)

        if index == 2:   # Sepia
            r = arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189
            g = arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168
            b = arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131
            return Image.fromarray(
                np.clip(np.stack([r,g,b],2), 0, 255).astype(np.uint8))

        if index == 3:   # Cool Mist
            arr[:,:,0] = np.clip(arr[:,:,0] * 0.80, 0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2] * 1.25 + 15, 0, 255)
            return ImageEnhance.Brightness(
                Image.fromarray(arr.astype(np.uint8))).enhance(0.95)

        if index == 4:   # Golden Hour
            arr[:,:,0] = np.clip(arr[:,:,0] * 1.20 + 20, 0, 255)
            arr[:,:,1] = np.clip(arr[:,:,1] * 1.05 + 5,  0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2] * 0.70,       0, 255)
            return Image.fromarray(arr.astype(np.uint8))

        if index == 5:   # Noir
            grey = ImageOps.grayscale(img)
            return ImageEnhance.Contrast(grey).enhance(1.6).convert("RGB")

        if index == 6:   # Faded
            img2 = ImageEnhance.Contrast(img).enhance(0.65)
            img2 = ImageEnhance.Color(img2).enhance(0.6)
            a2   = np.clip(np.array(img2, dtype=np.float64) * 0.85 + 30, 0, 255)
            return Image.fromarray(a2.astype(np.uint8))

        if index == 7:   # Cinematic
            a2 = arr.copy()
            dark = arr.mean(axis=2) < 100
            a2[dark, 0] = np.clip(arr[dark, 0] * 0.75, 0, 255)
            a2[dark, 2] = np.clip(arr[dark, 2] * 1.30, 0, 255)
            lite = arr.mean(axis=2) >= 150
            a2[lite, 0] = np.clip(arr[lite, 0] * 1.15 + 15, 0, 255)
            a2[lite, 2] = np.clip(arr[lite, 2] * 0.70,       0, 255)
            return ImageEnhance.Contrast(
                Image.fromarray(a2.astype(np.uint8))).enhance(1.15)

        if index == 8:   # Neon Pop
            img2 = ImageEnhance.Color(img).enhance(2.5)
            return ImageEnhance.Contrast(img2).enhance(1.4)

        if index == 9:   # Soft Bloom
            blur = img.filter(ImageFilter.GaussianBlur(radius=6))
            glow = np.clip(arr * 0.70 + np.array(blur, np.float64) * 0.50, 0, 255)
            return ImageEnhance.Brightness(
                Image.fromarray(glow.astype(np.uint8))).enhance(1.1)

        if index == 10:  # Deep Forest
            arr[:,:,0] = np.clip(arr[:,:,0] * 0.78,       0, 255)
            arr[:,:,1] = np.clip(arr[:,:,1] * 1.18 + 5,   0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2] * 0.85,       0, 255)
            return ImageEnhance.Contrast(
                Image.fromarray(arr.astype(np.uint8))).enhance(1.2)

        return img


# ─────────────────────────────────────────────────────────────────
# REGION BLUR ENGINE  — pixelate / gaussian blur on a drawn rect
# ─────────────────────────────────────────────────────────────────

class RegionBlurEngine:
    @staticmethod
    def pixelate(pil_img, x1, y1, x2, y2, block_size=12):
        """Classic censorship pixelation (mosaic / news-style)."""
        img = pil_img.copy()
        region = img.crop((x1, y1, x2, y2))
        w, h = region.size
        if w < 1 or h < 1:
            return img
        bs = max(2, block_size)
        small = region.resize((max(1, w // bs), max(1, h // bs)), Image.NEAREST)
        pixelated = small.resize((w, h), Image.NEAREST)
        img.paste(pixelated, (x1, y1))
        return img

    @staticmethod
    def gaussian_blur(pil_img, x1, y1, x2, y2, radius=18):
        """Smooth Gaussian blur on selected region."""
        img = pil_img.copy()
        region = img.crop((x1, y1, x2, y2))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=max(1, radius)))
        img.paste(blurred, (x1, y1))
        return img

    @staticmethod
    def heavy_blur(pil_img, x1, y1, x2, y2):
        """Layered blur — almost completely unrecognisable."""
        img = pil_img.copy()
        region = img.crop((x1, y1, x2, y2))
        w, h = region.size
        if w < 1 or h < 1:
            return img
        # Downsample → upsample → gaussian
        tiny = region.resize((max(1, w // 8), max(1, h // 8)), Image.NEAREST)
        up   = tiny.resize((w, h), Image.NEAREST)
        final = up.filter(ImageFilter.GaussianBlur(radius=15))
        img.paste(final, (x1, y1))
        return img


# ─────────────────────────────────────────────────────────────────
# BACKGROUND REMOVAL ENGINE  — polished multi-method
# ─────────────────────────────────────────────────────────────────

class BackgroundRemovalEngine:

    @staticmethod
    def by_color(pil_img, tolerance=40):
        """
        Flood-fill style: sample all four image corners, average their
        colour, then keep pixels whose Euclidean distance to that colour
        exceeds `tolerance`.  A morphological clean-up pass removes
        isolated specks.  Works best on plain / studio backgrounds.
        """
        img  = pil_img.convert("RGBA")
        arr  = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Sample a small border region (top, bottom, left, right strips)
        border = np.concatenate([
            arr[0,   :,  :3].reshape(-1,3),
            arr[h-1, :,  :3].reshape(-1,3),
            arr[:,   0,  :3].reshape(-1,3),
            arr[:, w-1,  :3].reshape(-1,3),
        ], axis=0)
        bg_col = np.median(border, axis=0)   # median is more robust than mean

        # Per-pixel Euclidean distance in RGB space
        dist = np.sqrt(np.sum((arr[:,:,:3] - bg_col)**2, axis=2))

        # Soft mask: ramp between tolerance*0.6 and tolerance
        lo, hi = tolerance * 0.6, float(tolerance)
        alpha  = np.clip((dist - lo) / max(hi - lo, 1), 0, 1)

        # Morphological clean: remove tiny foreground specks
        alpha_u8  = (alpha * 255).astype(np.uint8)
        mask_img  = Image.fromarray(alpha_u8, "L")
        # Erode then dilate (open) to kill isolated pixels
        mask_img  = mask_img.filter(ImageFilter.MinFilter(3))
        mask_img  = mask_img.filter(ImageFilter.MaxFilter(3))
        # Smooth edges
        mask_img  = mask_img.filter(ImageFilter.GaussianBlur(1))

        out       = np.array(img)
        out[:,:,3] = np.array(mask_img)
        return Image.fromarray(out, "RGBA")

    @staticmethod
    def by_threshold(pil_img, threshold=200, invert=False):
        """
        Luminance-based mask.  Pixels brighter than `threshold` are
        treated as background.  Tick `invert` for dark backgrounds
        (keep light subjects).  Includes edge-smoothing pass.
        """
        img  = pil_img.convert("RGBA")
        grey = np.array(pil_img.convert("L"), dtype=np.float32)

        if invert:
            # keep bright pixels (dark background)
            alpha = np.clip((grey - (255 - threshold)) / max(threshold, 1), 0, 1)
        else:
            # keep dark pixels (bright background)
            alpha = np.clip(1.0 - (grey / max(threshold, 1)), 0, 1)

        alpha_u8 = (alpha * 255).astype(np.uint8)
        mask_img = Image.fromarray(alpha_u8, "L")
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(1.5))

        out       = np.array(img)
        out[:,:,3] = np.array(mask_img)
        return Image.fromarray(out, "RGBA")

    @staticmethod
    def by_edge(pil_img, blur_r=2, dilate=8):
        """
        Detect strong edges, dilate the edge map to form a hull around
        the subject, then flood-fill the hull interior to create a
        foreground mask.  Best for objects with a clear outline.
        """
        grey  = pil_img.convert("L")
        blurred = grey.filter(ImageFilter.GaussianBlur(blur_r))
        edges = blurred.filter(ImageFilter.FIND_EDGES)

        # Boost contrast of edge map
        e_arr = np.array(edges, dtype=np.float32)
        e_arr = np.clip(e_arr / max(e_arr.max(), 1) * 255, 0, 255).astype(np.uint8)
        edges = Image.fromarray(e_arr, "L").point(lambda p: 255 if p > 20 else 0)

        # Dilate: grow the edge hull
        grown = edges
        for _ in range(dilate):
            grown = grown.filter(ImageFilter.MaxFilter(3))

        # Smooth mask and feather edges
        grown = grown.filter(ImageFilter.GaussianBlur(2))

        arr        = np.array(pil_img.convert("RGBA"))
        arr[:,:,3] = np.array(grown)
        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def by_chroma(pil_img, hue_center=120, hue_range=40, tolerance=60):
        """
        Chroma-key (green/blue screen) removal.
        hue_center: dominant BG hue in degrees (120=green, 240=blue).
        Pixels within hue_range of that hue AND saturation > tolerance
        are made transparent.
        """
        img  = pil_img.convert("RGBA")
        hsv  = pil_img.convert("RGB")
        arr  = np.array(hsv, dtype=np.float32) / 255.0
        # Convert RGB → HSV manually
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mx   = np.max(arr, axis=2)
        mn   = np.min(arr, axis=2)
        df   = mx - mn + 1e-9
        # Hue
        h_ch = np.zeros_like(mx)
        m    = mx == r;  h_ch[m] = (60*((g[m]-b[m])/df[m]) % 360)
        m    = mx == g;  h_ch[m] = (60*((b[m]-r[m])/df[m]) + 120)
        m    = mx == b;  h_ch[m] = (60*((r[m]-g[m])/df[m]) + 240)
        # Saturation
        s_ch = df / (mx + 1e-9)

        hue_dist = np.abs(h_ch - hue_center)
        hue_dist = np.minimum(hue_dist, 360 - hue_dist)
        is_chroma = (hue_dist < hue_range) & (s_ch * 255 > tolerance)

        out        = np.array(img)
        out[:,:,3] = np.where(is_chroma, 0, 255).astype(np.uint8)
        # Feather edges
        mask_img   = Image.fromarray(out[:,:,3], "L").filter(
            ImageFilter.GaussianBlur(1))
        out[:,:,3] = np.array(mask_img)
        return Image.fromarray(out, "RGBA")

    @staticmethod
    def replace_background(fg_rgba, bg_color=(0,0,0)):
        """Flatten RGBA onto a solid colour (for JPEG-compatible output)."""
        bg = Image.new("RGB", fg_rgba.size, bg_color)
        bg.paste(fg_rgba, mask=fg_rgba.split()[3])
        return bg


# ─────────────────────────────────────────────────────────────────
# STEGANOGRAPHY ENGINE  — spatial-domain LSB  O(N)  vectorised
#
# HOW IT WORKS — ENCODING
#   Every character in your message is converted to its 8-bit ASCII
#   binary value, e.g. 'H' → 01001000.  Those bits are written one
#   per pixel into the LEAST SIGNIFICANT BIT of the Red channel.
#   Changing 1 bit in 255 causes a colour shift of ±1 — completely
#   invisible to the human eye.  A 7-character delimiter "<<END>>"
#   (56 extra bits) is appended so we know where to stop reading.
#   The whole operation is a single numpy vectorised pass → O(N).
#
# HOW IT WORKS — DECODING
#   Read the LSB of every Red pixel with a bitwise AND (& 1).
#   Pack every 8 consecutive bits back into a byte with np.packbits.
#   Convert bytes → string, stop the moment we hit "<<END>>".
#   Again fully vectorised — no Python loops over pixels.
#
# COMPLEXITY
#   hide  : O(N)  — one numpy mask + one assignment over N pixels
#   reveal: O(N)  — one np.packbits call, then a single string search
# ─────────────────────────────────────────────────────────────────

class SteganographyEngine:
    # 8-byte binary sentinel — cannot appear by accident in UTF-8 text
    _DELIM = b"\x00\xFF\x00\xFF\xAA\x55\xDE\xAD"

    # ── public API ───────────────────────────────────────────────
    @staticmethod
    def capacity(pil_img) -> int:
        """Maximum message bytes this image can carry."""
        h, w = np.array(pil_img).shape[:2]
        return (h * w) // 8 - len(SteganographyEngine._DELIM) - 1

    @staticmethod
    def hide(pil_img, message: str) -> Image.Image:
        """
        Encode message into the LSBs of the Red channel (spatial domain).
        Uses flat[0::3] strided view so writes propagate back to arr.
        Complexity: O(N) — fully vectorised.
        """
        arr  = np.ascontiguousarray(
                   np.array(pil_img.convert("RGB"), dtype=np.uint8))
        # Red pixels live at every 3rd byte in the flat C-contiguous array
        red  = arr.ravel()[0::3]          # strided view — writes go to arr
        bits = np.unpackbits(
                   np.frombuffer(
                       message.encode("utf-8") + SteganographyEngine._DELIM,
                       dtype=np.uint8))
        n = len(bits)
        if n > len(red):
            cap = len(red) // 8 - len(SteganographyEngine._DELIM) - 1
            raise ValueError(
                f"Message too long!\n"
                f"  Needs   : {n} bits ({n//8} bytes)\n"
                f"  Capacity: {len(red)} bits ({cap} usable bytes)\n"
                f"Use a larger image or a shorter message.")
        # Single vectorised write — O(N)
        red[:n] = (red[:n] & np.uint8(0xFE)) | bits.astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    @staticmethod
    def reveal(pil_img) -> str:
        """
        Decode message from LSBs of the Red channel.
        Searches for sentinel in raw bytes — immune to UTF-8 decode errors.
        Complexity: O(N).
        """
        arr  = np.ascontiguousarray(
                   np.array(pil_img.convert("RGB"), dtype=np.uint8))
        red  = arr.ravel()[0::3]
        bits = (red & np.uint8(1)).astype(np.uint8)
        n    = (len(bits) // 8) * 8
        raw  = np.packbits(bits[:n]).tobytes()
        idx  = raw.find(SteganographyEngine._DELIM)
        if idx == -1:
            return ""    # no sentinel → no hidden message
        return raw[:idx].decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────
# WATERMARK ENGINE  — text & image watermark
# ─────────────────────────────────────────────────────────────────

class WatermarkEngine:

    @staticmethod
    def text(pil_img, text, position="bottom-right",
             opacity=0.55, font_scale=1.0, color=(255,255,255)):
        """Stamp semi-transparent text onto image."""
        from PIL import ImageDraw, ImageFont
        img  = pil_img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        size = max(12, int(min(w, h) * 0.04 * font_scale))
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            except Exception:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pad = int(size * 0.6)
        positions = {
            "top-left":     (pad, pad),
            "top-right":    (w - tw - pad, pad),
            "bottom-left":  (pad, h - th - pad),
            "bottom-right": (w - tw - pad, h - th - pad),
            "center":       ((w - tw)//2, (h - th)//2),
        }
        x, y = positions.get(position, positions["bottom-right"])
        a    = int(opacity * 255)
        # Shadow
        draw.text((x+2, y+2), text, font=font, fill=(0,0,0,max(0,a-60)))
        draw.text((x,   y),   text, font=font, fill=(*color, a))
        return Image.alpha_composite(img, overlay).convert("RGB")

    @staticmethod
    def tiled(pil_img, text, opacity=0.18, font_scale=0.8, color=(200,200,200)):
        """Diagonal repeating tile watermark (like Google Docs)."""
        from PIL import ImageDraw, ImageFont
        import math
        img = pil_img.convert("RGBA")
        w, h = img.size
        size = max(10, int(min(w,h) * 0.035 * font_scale))
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except Exception:
                font = ImageFont.load_default()

        overlay = Image.new("RGBA", (w, h), (0,0,0,0))
        draw    = ImageDraw.Draw(overlay)
        a       = int(opacity * 255)
        step_x, step_y = int(w * 0.22), int(h * 0.15)
        for row in range(-1, int(h/step_y)+2):
            for col in range(-1, int(w/step_x)+2):
                x = col * step_x + (row % 2) * (step_x // 2)
                y = row * step_y
                draw.text((x, y), text, font=font, fill=(*color, a))
        rotated = overlay.rotate(25, expand=False)
        return Image.alpha_composite(img, rotated).convert("RGB")

    @staticmethod
    def image_stamp(base_img, stamp_path, position="bottom-right", opacity=0.6, scale=0.15):
        """Paste a logo/image as watermark."""
        base  = base_img.convert("RGBA")
        stamp = Image.open(stamp_path).convert("RGBA")
        bw, bh = base.size
        sw = max(16, int(bw * scale))
        ratio = sw / stamp.width
        sh    = max(1, int(stamp.height * ratio))
        stamp = stamp.resize((sw, sh), Image.LANCZOS)
        # Apply opacity
        r,g,b,a = stamp.split()
        a = a.point(lambda p: int(p * opacity))
        stamp.putalpha(a)
        pad = 12
        positions = {
            "top-left":     (pad, pad),
            "top-right":    (bw - sw - pad, pad),
            "bottom-left":  (pad, bh - sh - pad),
            "bottom-right": (bw - sw - pad, bh - sh - pad),
            "center":       ((bw - sw)//2, (bh - sh)//2),
        }
        xy = positions.get(position, positions["bottom-right"])
        base.paste(stamp, xy, mask=stamp)
        return base.convert("RGB")


# ─────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────

class FourierImageStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fourier Image Studio")
        self.geometry("1340x860")
        self.minsize(1100, 720)
        self.configure(bg=P["bg"])

        self.original_image  = None
        self.processed_image = None
        self.image_path      = None
        self._orig_tk        = None
        self._proc_tk        = None
        self._processing     = False
        self._refreshing     = False   # guard against recursive Configure events
        self._filter_thumbs  = []
        self._active_filter  = tk.IntVar(value=0)
        self._active_tab     = tk.StringVar(value="sharpen")
        self.strength_var    = tk.DoubleVar(value=0.65)

        # ── Region-blur state ────────────────────
        self._blur_rects      = []
        self._blur_mode       = tk.StringVar(value="pixelate")
        self._blur_block      = tk.IntVar(value=15)
        self._blur_radius     = tk.IntVar(value=20)
        self._rb_start        = None
        self._rb_rect_id      = None
        self._rb_active       = False

        # ── Background removal state ─────────────
        self._bg_method       = tk.StringVar(value="color")
        self._bg_tolerance    = tk.IntVar(value=40)
        self._bg_threshold    = tk.IntVar(value=200)
        self._bg_color_r      = tk.IntVar(value=0)
        self._bg_color_g      = tk.IntVar(value=0)
        self._bg_color_b      = tk.IntVar(value=0)
        self._bg_hue_var      = tk.IntVar(value=120)
        self._bg_invert_var   = tk.BooleanVar(value=False)

        # ── Steganography state ───────────────────
        self._steg_message    = tk.StringVar(value="")
        self._steg_revealed   = tk.StringVar(value="")

        # ── Watermark state ──────────────────────
        self._wm_text         = tk.StringVar(value="© My Studio")
        self._wm_mode         = tk.StringVar(value="single")
        self._wm_position     = tk.StringVar(value="bottom-right")
        self._wm_opacity      = tk.DoubleVar(value=0.55)
        self._wm_font_scale   = tk.DoubleVar(value=1.0)
        self._wm_color_hex    = tk.StringVar(value="#ffffff")
        self._wm_stamp_path   = tk.StringVar(value="")

        self._build_styles()
        self._build_ui()
        self._bind_events()

    # ── STYLES ────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",            background=P["bg"])
        s.configure("TLabel",            background=P["bg"],
                    foreground=P["text"], font=FONT_LABEL)
        s.configure("Horizontal.TScale", background=P["card"],
                    troughcolor=P["border"], sliderthickness=16,
                    sliderrelief="flat")

    # ── ROOT UI ───────────────────────────────────────────────────
    def _build_ui(self):
        self._build_topbar()
        body = tk.Frame(self, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=14, pady=(10, 0))
        self._build_sidebar(body)
        self._build_viewer(body)
        self._build_statusbar()

    # ── TOP BAR ───────────────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self, bg=P["surface"], height=58)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡", bg=P["surface"], fg=P["accent"],
                 font=("Segoe UI", 20)).pack(side="left", padx=(16, 6))
        tk.Label(bar, text="FOURIER IMAGE STUDIO |Charles Kabunga",
                 bg=P["surface"], fg=P["white"],
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        tk.Label(bar, text="  ·  FFT Enhancement + Live Colour Filters",
                 bg=P["surface"], fg=P["muted"], font=FONT_SUB).pack(side="left")
        for txt, cmd, col in [("💾  Save", self._save_image, P["accent2"]),
                               ("📂  Open", self._open_image, P["accent"])]:
            tk.Button(bar, text=txt, command=cmd,
                      bg=col, fg=P["white"], activebackground=P["bg"],
                      activeforeground=P["white"], font=FONT_BTN,
                      relief="flat", padx=14, pady=0, cursor="hand2"
                      ).pack(side="right", padx=(0, 10), pady=10)

    # ── STATUS BAR ────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=P["surface"], height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready — open an image to begin.")
        tk.Label(bar, textvariable=self.status_var, bg=P["surface"],
                 fg=P["muted"], font=FONT_SUB, anchor="w"
                 ).pack(side="left", padx=12, pady=3)
        tk.Label(bar, text="Algorithm: 2D FFT (Cooley-Tukey)  ·  O(NM log NM)",
                 bg=P["surface"], fg=P["accent"], font=FONT_SUB
                 ).pack(side="right", padx=12, pady=3)

    # ── SIDEBAR ───────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=P["card"],
                        highlightbackground=P["border"], highlightthickness=1,
                        width=295)
        side.pack(side="left", fill="y", padx=(0, 12), pady=(0, 12))
        side.pack_propagate(False)

        # Logo
        tk.Label(side, text="⬡", bg=P["card"], fg=P["accent"],
                 font=("Segoe UI", 30)).pack(pady=(18, 0))
        tk.Label(side, text="Image Processing", bg=P["card"],
                 fg=P["white"], font=("Segoe UI", 11, "bold")).pack()
        tk.Label(side, text="via Fourier Series", bg=P["card"],
                 fg=P["muted"], font=FONT_SUB).pack(pady=(0, 8))

        # ── TAB ROWS ──────────────────────────
        self._tab_btns = {}

        # ── ROW 1: Core processing tabs ───────────
        row1_outer = tk.Frame(side, bg=P["border"], pady=1, padx=1)
        row1_outer.pack(fill="x", padx=14, pady=(4, 0))
        row1_inner = tk.Frame(row1_outer, bg=P["border"])
        row1_inner.pack(fill="x")

        row1_tabs = [("sharpen","◈ Sharpen"), ("smooth","◉ Smooth"),
                     ("filters","✦ Filters"), ("blur","⬛ Blur"), ("info","ⓘ Info")]
        for val, label in row1_tabs:
            btn = tk.Button(row1_inner, text=label,
                bg=P["card"], fg=P["muted"],
                activebackground=P["accent"], activeforeground=P["white"],
                font=("Segoe UI", 8, "bold"), relief="flat",
                padx=0, pady=7, cursor="hand2",
                command=lambda v=val: self._switch_tab(v))
            btn.pack(side="left", fill="x", expand=True, padx=1)
            self._tab_btns[val] = btn

        # ── ROW 2: Creative / utility tabs ────────
        row2_outer = tk.Frame(side, bg=P["border"], pady=1, padx=1)
        row2_outer.pack(fill="x", padx=14, pady=(2, 0))
        row2_inner = tk.Frame(row2_outer, bg=P["border"])
        row2_inner.pack(fill="x")

        row2_tabs = [("bgremove","✂ BG Remove"), ("steg","🔒 Steganography"),
                     ("watermark","💧 Watermark")]
        for val, label in row2_tabs:
            btn = tk.Button(row2_inner, text=label,
                bg=P["surface"], fg=P["muted"],
                activebackground="#2a4a7a", activeforeground=P["white"],
                font=("Segoe UI", 8, "bold"), relief="flat",
                padx=0, pady=7, cursor="hand2",
                command=lambda v=val: self._switch_tab(v))
            btn.pack(side="left", fill="x", expand=True, padx=1)
            self._tab_btns[val] = btn

        # ── SCROLLABLE TAB CONTENT AREA ───────────
        scroll_outer = tk.Frame(side, bg=P["card"])
        scroll_outer.pack(fill="both", expand=True, pady=(4, 0))

        self._side_canvas = tk.Canvas(scroll_outer, bg=P["card"],
                                      highlightthickness=0, bd=0)
        self._side_canvas.pack(side="left", fill="both", expand=True)

        _vsb = tk.Scrollbar(scroll_outer, orient="vertical",
                            command=self._side_canvas.yview,
                            width=6, troughcolor=P["surface"],
                            bg=P["border"], relief="flat",
                            activebackground=P["accent"])
        _vsb.pack(side="right", fill="y")
        self._side_canvas.configure(yscrollcommand=_vsb.set)

        self._tab_frame = tk.Frame(self._side_canvas, bg=P["card"])
        self._tab_frame_id = self._side_canvas.create_window(
            (0, 0), window=self._tab_frame, anchor="nw")

        def _on_canvas_resize(e):
            self._side_canvas.itemconfig(self._tab_frame_id, width=e.width)
        self._side_canvas.bind("<Configure>", _on_canvas_resize)

        def _on_frame_resize(e):
            self._side_canvas.configure(
                scrollregion=self._side_canvas.bbox("all"))
        self._tab_frame.bind("<Configure>", _on_frame_resize)

        def _on_mousewheel(e):
            if e.delta:
                self._side_canvas.yview_scroll(int(-e.delta / 60), "units")
            elif e.num == 4:
                self._side_canvas.yview_scroll(-2, "units")
            elif e.num == 5:
                self._side_canvas.yview_scroll(2, "units")

        for _w in (self._side_canvas, self._tab_frame):
            _w.bind("<MouseWheel>", _on_mousewheel)
            _w.bind("<Button-4>",   _on_mousewheel)
            _w.bind("<Button-5>",   _on_mousewheel)

        def _bind_scroll_children(widget):
            try:
                widget.bind("<MouseWheel>", _on_mousewheel, add="+")
                widget.bind("<Button-4>",   _on_mousewheel, add="+")
                widget.bind("<Button-5>",   _on_mousewheel, add="+")
            except Exception:
                pass
            for child in widget.winfo_children():
                _bind_scroll_children(child)

        self._bind_scroll_children = _bind_scroll_children
        self._scroll_to_top = lambda: self._side_canvas.yview_moveto(0)

        self._build_tab_sharpen()
        self._build_tab_smooth()
        self._build_tab_filters()
        self._build_tab_blur()
        self._build_tab_bgremove()
        self._build_tab_steg()
        self._build_tab_watermark()
        self._build_tab_info()
        # Bind mousewheel on every widget in the sidebar content area
        self.after(200, lambda: self._bind_scroll_children(self._tab_frame))
        self._switch_tab("sharpen")

    def _switch_tab(self, val):
        self._active_tab.set(val)
        ROW2 = {"bgremove", "steg", "watermark"}
        for k, btn in self._tab_btns.items():
            if k == val:
                active_bg = "#2a4a7a" if k in ROW2 else P["accent"]
                btn.config(bg=active_bg, fg=P["white"])
            else:
                idle_bg = P["surface"] if k in ROW2 else P["card"]
                btn.config(bg=idle_bg, fg=P["muted"])
        for w in self._tab_frame.winfo_children():
            w.pack_forget()
        panel = getattr(self, f"_panel_{val}", None)
        if panel:
            panel.pack(fill="both", expand=True)
            # Rebind scroll on all newly visible widgets
            try:
                self._bind_scroll_children(panel)
            except Exception:
                pass
        # Always jump back to the top of the content area
        try:
            self._scroll_to_top()
        except Exception:
            pass

    # ── TAB: SHARPEN ──────────────────────────
    def _build_tab_sharpen(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_sharpen = p
        self._section(p, "SHARPEN MODE")
        tk.Label(p, text="FFT unsharp mask + Wiener deconvolution\nfor artefact-free edge clarity.",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB, justify="left",
                 wraplength=250).pack(anchor="w", padx=16, pady=(2, 8))
        self._strength_block(p)
        self._presets_block(p)
        self._section(p, "ACTIONS", P["accent2"])
        f = tk.Frame(p, bg=P["card"])
        f.pack(fill="x", padx=14, pady=(8, 0))
        self._sharpen_btn = tk.Button(f, text="▶  SHARPEN IMAGE",
            bg=P["accent"], fg=P["white"], activebackground="#5a52e0",
            font=FONT_BTN, relief="flat", pady=10, cursor="hand2",
            command=lambda: self._run_fft("sharpen"))
        self._sharpen_btn.pack(fill="x", pady=(0, 6))
        self._reset_btn_for(f)

    # ── TAB: SMOOTH ───────────────────────────
    def _build_tab_smooth(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_smooth = p
        self._section(p, "SMOOTH / BLUR MODE")
        tk.Label(p, text="Gaussian low-pass filter in frequency domain.\nHigher strength = more blur.",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB, justify="left",
                 wraplength=250).pack(anchor="w", padx=16, pady=(2, 8))
        self._strength_block(p)
        self._presets_block(p)
        self._section(p, "ACTIONS", P["accent2"])
        f = tk.Frame(p, bg=P["card"])
        f.pack(fill="x", padx=14, pady=(8, 0))
        self._smooth_btn = tk.Button(f, text="▶  SMOOTH IMAGE",
            bg=P["accent"], fg=P["white"], activebackground="#5a52e0",
            font=FONT_BTN, relief="flat", pady=10, cursor="hand2",
            command=lambda: self._run_fft("smooth"))
        self._smooth_btn.pack(fill="x", pady=(0, 6))
        self._reset_btn_for(f)

    # ── TAB: FILTERS ──────────────────────────
    def _build_tab_filters(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_filters = p
        self._section(p, "COLOUR FILTERS")
        tk.Label(p, text="Scroll ← → through filters  ·  Click to apply instantly",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB
                 ).pack(anchor="w", padx=16, pady=(2, 6))

        # Carousel
        co = tk.Frame(p, bg=P["border"], pady=1, padx=1)
        co.pack(fill="x", padx=14, pady=(0, 6))
        ci = tk.Frame(co, bg=P["surface"])
        ci.pack(fill="x")

        self._fc = tk.Canvas(ci, bg=P["surface"],
                             highlightthickness=0, height=128)
        self._fc.pack(fill="x", expand=True)
        hscroll = tk.Scrollbar(ci, orient="horizontal", command=self._fc.xview)
        hscroll.pack(fill="x")
        self._fc.configure(xscrollcommand=hscroll.set)

        self._frow = tk.Frame(self._fc, bg=P["surface"])
        self._fc.create_window((0, 0), window=self._frow, anchor="nw")
        self._frow.bind("<Configure>", lambda e:
            self._fc.configure(scrollregion=self._fc.bbox("all")))

        # Mouse-wheel / trackpad horizontal scroll
        def _hscroll(e):
            delta = -1 * (e.delta // 120) if e.delta else (1 if e.num == 5 else -1)
            self._fc.xview_scroll(delta, "units")
        self._fc.bind("<MouseWheel>", _hscroll)
        self._fc.bind("<Button-4>",   _hscroll)
        self._fc.bind("<Button-5>",   _hscroll)

        self._filter_lbl_var = tk.StringVar(value="○  Original  —  No filter applied")

        self._build_filter_tiles()

        # Selected filter label
        tk.Label(p, textvariable=self._filter_lbl_var, bg=P["card"],
                 fg=P["accent2"], font=("Segoe UI", 9, "bold"), wraplength=255
                 ).pack(anchor="w", padx=16, pady=(0, 6))

        # Apply / Reset
        self._section(p, "ACTIONS", P["accent2"])
        f = tk.Frame(p, bg=P["card"])
        f.pack(fill="x", padx=14, pady=(8, 0))
        self.filter_apply_btn = tk.Button(f, text="▶  APPLY FILTER",
            bg=P["accent"], fg=P["white"], activebackground="#5a52e0",
            font=FONT_BTN, relief="flat", pady=10, cursor="hand2",
            command=self._run_filter)
        self.filter_apply_btn.pack(fill="x", pady=(0, 6))
        self._reset_btn_for(f)

    def _build_filter_tiles(self):
        TW, TH = 82, 70
        self._tiles = []
        for i, (name, desc, icon) in enumerate(FilterEngine.FILTERS):
            tile = tk.Frame(self._frow, bg=P["surface"],
                            highlightbackground=P["border"],
                            highlightthickness=1, cursor="hand2")
            tile.pack(side="left", padx=4, pady=6)
            c = tk.Canvas(tile, width=TW, height=TH, bg=P["card"],
                          highlightthickness=0)
            c.pack()
            c.create_text(TW//2, TH//2, text=icon,
                          fill=P["muted"], font=("Segoe UI", 16))
            lbl = tk.Label(tile, text=name, bg=P["surface"], fg=P["muted"],
                           font=("Segoe UI", 7, "bold"), width=10)
            lbl.pack(pady=(1, 4))
            idx = i
            for w in (tile, c, lbl):
                w.bind("<Button-1>", lambda e, n=idx: self._select_filter(n))
            self._tiles.append((tile, c, lbl))
        self._select_filter(0, apply=False)

    def _select_filter(self, idx, apply=True):
        self._active_filter.set(idx)
        name, desc, icon = FilterEngine.FILTERS[idx]
        self._filter_lbl_var.set(f"{icon}  {name}  —  {desc}")
        for i, (tile, c, lbl) in enumerate(self._tiles):
            if i == idx:
                tile.config(highlightbackground=P["accent"],
                            highlightthickness=2)
                lbl.config(fg=P["accent2"])
            else:
                tile.config(highlightbackground=P["border"],
                            highlightthickness=1)
                lbl.config(fg=P["muted"])
        if apply and self.original_image:
            self._run_filter()

    def _populate_filter_thumbs(self):
        """Render thumbnails: heavy PIL work in background, Tk calls on main thread."""
        if not self.original_image:
            return
        TW, TH = 82, 70
        small = self.original_image.copy()
        small.thumbnail((TW * 2, TH * 2), Image.BILINEAR)

        def worker():
            results = []
            for i in range(len(FilterEngine.FILTERS)):
                try:
                    result = FilterEngine.apply(small, i)
                    result.thumbnail((TW, TH), Image.BILINEAR)
                    results.append((i, result))
                except Exception:
                    results.append((i, None))
            # Hand off ALL Tk/PhotoImage work to the main thread at once
            self.after(0, lambda: self._apply_filter_thumbs(results, TW, TH))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_filter_thumbs(self, results, TW, TH):
        """Called on the main thread — safe to create ImageTk.PhotoImage here."""
        self._filter_thumbs = [None] * len(FilterEngine.FILTERS)
        for i, pil_img in results:
            if pil_img is None:
                continue
            try:
                tk_img = ImageTk.PhotoImage(pil_img)
                self._filter_thumbs[i] = tk_img
                tile, c, lbl = self._tiles[i]
                c.delete("all")
                c.create_image(TW // 2, TH // 2, image=tk_img, anchor="center")
            except Exception:
                pass

    # ── TAB: REGION BLUR ──────────────────────
    def _build_tab_blur(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_blur = p

        self._section(p, "REGION BLUR / CENSOR")
        tk.Label(p, text="Draw a box on the LEFT image.\nThe selected area will be censored.",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB, justify="left",
                 wraplength=250).pack(anchor="w", padx=16, pady=(2, 8))

        # ── Blur method radio buttons ─────────
        self._section(p, "BLUR METHOD")
        mf = tk.Frame(p, bg=P["card"])
        mf.pack(fill="x", padx=14, pady=(6, 0))
        for txt, val in [("Pixelate  (news / mosaic)", "pixelate"),
                          ("Gaussian  (smooth fog)",   "gaussian"),
                          ("Heavy     (max censor)",   "heavy")]:
            tk.Radiobutton(mf, text=txt, variable=self._blur_mode, value=val,
                bg=P["card"], fg=P["text"], selectcolor=P["surface"],
                activebackground=P["card"], activeforeground=P["accent"],
                font=("Segoe UI", 8), anchor="w"
            ).pack(fill="x", pady=1)

        # ── Pixelate block-size slider ─────────
        self._section(p, "PIXELATE BLOCK SIZE")
        bf = tk.Frame(p, bg=P["card"])
        bf.pack(fill="x", padx=14, pady=(6, 0))
        self._blur_block_disp = tk.Label(bf, text="15 px", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI", 14, "bold"))
        self._blur_block_disp.pack()
        ttk.Scale(bf, from_=4, to=40, variable=self._blur_block,
                  orient="horizontal",
                  command=lambda v: self._blur_block_disp.config(
                      text=f"{int(float(v))} px")
                  ).pack(fill="x")

        # ── Gaussian radius slider ─────────────
        self._section(p, "GAUSSIAN RADIUS")
        rf = tk.Frame(p, bg=P["card"])
        rf.pack(fill="x", padx=14, pady=(6, 0))
        self._blur_radius_disp = tk.Label(rf, text="20 px", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI", 14, "bold"))
        self._blur_radius_disp.pack()
        ttk.Scale(rf, from_=2, to=60, variable=self._blur_radius,
                  orient="horizontal",
                  command=lambda v: self._blur_radius_disp.config(
                      text=f"{int(float(v))} px")
                  ).pack(fill="x")

        # ── Rect list + clear ──────────────────
        self._section(p, "DRAWN REGIONS", P["warn"])
        self._blur_count_var = tk.StringVar(value="No regions drawn yet")
        tk.Label(p, textvariable=self._blur_count_var, bg=P["card"],
                 fg=P["warn"], font=FONT_SUB, wraplength=255
                 ).pack(anchor="w", padx=16, pady=(4, 2))

        # ── Actions ───────────────────────────
        self._section(p, "ACTIONS", P["accent2"])
        af = tk.Frame(p, bg=P["card"])
        af.pack(fill="x", padx=14, pady=(8, 0))

        self._blur_apply_btn = tk.Button(af, text="▶  APPLY BLUR REGIONS",
            bg=P["accent"], fg=P["white"], activebackground="#5a52e0",
            font=FONT_BTN, relief="flat", pady=10, cursor="hand2",
            command=self._run_region_blur)
        self._blur_apply_btn.pack(fill="x", pady=(0, 4))

        tk.Button(af, text="✕  CLEAR ALL REGIONS", bg=P["warn"], fg=P["white"],
            activebackground="#cc4444", font=FONT_BTN, relief="flat",
            pady=8, cursor="hand2", command=self._clear_blur_rects
        ).pack(fill="x", pady=(0, 4))

        self._reset_btn_for(af)

        tk.Label(p,
            text="① Click the ⬛ Blur tab\n"
                 "② Click & drag on the LEFT\n"
                 "   image to draw a region\n"
                 "③ Draw as many as needed\n"
                 "④ Choose blur method above\n"
                 "⑤ Press ▶ APPLY BLUR REGIONS",
            bg=P["surface"], fg=P["muted"], font=FONT_MONO,
            justify="left", padx=10, pady=10
        ).pack(fill="x", padx=16, pady=(10, 4))

    # ── TAB: BACKGROUND REMOVAL ───────────────
    def _build_tab_bgremove(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_bgremove = p

        self._section(p, "✂ BACKGROUND REMOVAL")
        tk.Label(p, text="4 methods — choose based on your image type.",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB,
                 justify="left", wraplength=250).pack(anchor="w", padx=16, pady=(2,6))

        # Method selector
        self._section(p, "METHOD")
        mf = tk.Frame(p, bg=P["card"])
        mf.pack(fill="x", padx=14, pady=(4,0))
        for txt, val, tip in [
            ("Corner Colour    ← plain/studio BG",  "color",     ""),
            ("Luminance Thresh ← bright or dark BG", "threshold", ""),
            ("Edge Hull        ← objects w/ outline","edge",      ""),
            ("Chroma Key       ← green / blue screen","chroma",   ""),
        ]:
            tk.Radiobutton(mf, text=txt, variable=self._bg_method, value=val,
                bg=P["card"], fg=P["text"], selectcolor=P["surface"],
                activebackground=P["card"], activeforeground=P["accent"],
                font=("Segoe UI", 7), anchor="w"
            ).pack(fill="x", pady=1)

        # Tolerance (corner colour)
        self._section(p, "COLOUR TOLERANCE  (Corner method)")
        tf = tk.Frame(p, bg=P["card"])
        tf.pack(fill="x", padx=14, pady=(2,0))
        self._bg_tol_disp = tk.Label(tf, text="40", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI",12,"bold"))
        self._bg_tol_disp.pack()
        ttk.Scale(tf, from_=5, to=150, variable=self._bg_tolerance,
                  orient="horizontal",
                  command=lambda v: self._bg_tol_disp.config(text=str(int(float(v))))
                  ).pack(fill="x")

        # Threshold + invert
        self._section(p, "LUMINANCE THRESHOLD  (Threshold method)")
        lf = tk.Frame(p, bg=P["card"])
        lf.pack(fill="x", padx=14, pady=(2,0))
        self._bg_lum_disp = tk.Label(lf, text="200", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI",12,"bold"))
        self._bg_lum_disp.pack()
        ttk.Scale(lf, from_=30, to=254, variable=self._bg_threshold,
                  orient="horizontal",
                  command=lambda v: self._bg_lum_disp.config(text=str(int(float(v))))
                  ).pack(fill="x")
        self._bg_invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(lf, text="Invert  (keep bright subjects on dark BG)",
            variable=self._bg_invert_var,
            bg=P["card"], fg=P["muted"], selectcolor=P["surface"],
            activebackground=P["card"], font=("Segoe UI",7)
        ).pack(anchor="w")

        # Chroma hue
        self._section(p, "CHROMA HUE °  (Chroma method)")
        hf = tk.Frame(p, bg=P["card"])
        hf.pack(fill="x", padx=14, pady=(2,0))
        self._bg_hue_var  = tk.IntVar(value=120)
        self._bg_hue_disp = tk.Label(hf, text="120° (Green)", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI",11,"bold"))
        self._bg_hue_disp.pack()
        def _hue_upd(v):
            deg = int(float(v))
            name = ("Red" if deg<30 or deg>330 else "Yellow" if deg<90
                    else "Green" if deg<150 else "Cyan" if deg<210
                    else "Blue" if deg<270 else "Magenta")
            self._bg_hue_disp.config(text=f"{deg}° ({name})")
        ttk.Scale(hf, from_=0, to=359, variable=self._bg_hue_var,
                  orient="horizontal", command=_hue_upd).pack(fill="x")

        # Replace BG colour
        self._section(p, "REPLACE BACKGROUND WITH COLOUR")
        cf = tk.Frame(p, bg=P["card"])
        cf.pack(fill="x", padx=14, pady=(4,0))
        self._bg_preview = tk.Label(cf, text="   ", bg="#000000", width=3)
        self._bg_preview.pack(side="left", padx=(0,5))
        for lbl, var in [("R", self._bg_color_r),
                          ("G", self._bg_color_g),
                          ("B", self._bg_color_b)]:
            tk.Label(cf, text=lbl, bg=P["card"], fg=P["muted"],
                     font=("Segoe UI",8)).pack(side="left")
            ttk.Scale(cf, from_=0, to=255, variable=var, orient="horizontal",
                      length=44,
                      command=lambda v, _=None: self._update_bg_preview()
                      ).pack(side="left", padx=(1,3))

        # Actions
        self._section(p, "ACTIONS", P["accent2"])
        af = tk.Frame(p, bg=P["card"])
        af.pack(fill="x", padx=14, pady=(6,0))
        self._bgr_btn = tk.Button(af, text="▶  REMOVE BACKGROUND",
            bg="#2a4a7a", fg=P["white"], activebackground="#1a3a6a",
            font=FONT_BTN, relief="flat", pady=9, cursor="hand2",
            command=self._run_bg_remove)
        self._bgr_btn.pack(fill="x", pady=(0,5))
        self._reset_btn_for(af)

    def _update_bg_preview(self):
        r = self._bg_color_r.get()
        g = self._bg_color_g.get()
        b = self._bg_color_b.get()
        self._bg_preview.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    # ── TAB: STEGANOGRAPHY ────────────────────
    def _build_tab_steg(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_steg = p

        self._section(p, "🔒 STEGANOGRAPHY")
        tk.Label(p, text="Hide or reveal a secret message\ninside an image (LSB encoding).",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB,
                 justify="left", wraplength=250).pack(anchor="w", padx=16, pady=(2,8))

        # ── HIDE ──────────────────────────────
        self._section(p, "HIDE MESSAGE IN IMAGE")
        hf = tk.Frame(p, bg=P["card"])
        hf.pack(fill="x", padx=14, pady=(6,0))
        tk.Label(hf, text="Secret message:", bg=P["card"],
                 fg=P["text"], font=FONT_LABEL).pack(anchor="w")
        self._steg_entry = tk.Text(hf, height=3, width=28,
            bg=P["surface"], fg=P["white"], insertbackground=P["white"],
            font=("Segoe UI", 9), relief="flat", wrap="word",
            highlightbackground=P["border"], highlightthickness=1)
        self._steg_entry.pack(fill="x", pady=(3,0))

        hbf = tk.Frame(p, bg=P["card"])
        hbf.pack(fill="x", padx=14, pady=(6,0))
        self._steg_hide_btn = tk.Button(hbf, text="🔒  HIDE MESSAGE",
            bg=P["accent"], fg=P["white"], activebackground="#5a52e0",
            font=FONT_BTN, relief="flat", pady=8, cursor="hand2",
            command=self._run_steg_hide)
        self._steg_hide_btn.pack(fill="x", pady=(0,4))

        # ── REVEAL ────────────────────────────
        self._section(p, "REVEAL HIDDEN MESSAGE")
        tk.Label(p, text="Open the carrier image first, then press reveal.",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB,
                 wraplength=250).pack(anchor="w", padx=16, pady=(4,4))
        rbf = tk.Frame(p, bg=P["card"])
        rbf.pack(fill="x", padx=14, pady=(0,0))
        self._steg_reveal_btn = tk.Button(rbf, text="🔍  REVEAL MESSAGE",
            bg="#2a5a4a", fg=P["white"], activebackground="#1a4a3a",
            font=FONT_BTN, relief="flat", pady=8, cursor="hand2",
            command=self._run_steg_reveal)
        self._steg_reveal_btn.pack(fill="x", pady=(0,4))

        # ── Output display area ───────────────
        self._section(p, "DECODED MESSAGE", P["accent2"])
        of = tk.Frame(p, bg=P["card"])
        of.pack(fill="x", padx=14, pady=(4, 0))

        # Header row: label + copy button
        hdr = tk.Frame(of, bg=P["card"])
        hdr.pack(fill="x", pady=(0, 3))
        tk.Label(hdr, text="Revealed message:", bg=P["card"],
                 fg=P["text"], font=FONT_LABEL).pack(side="left")
        self._steg_copy_btn = tk.Button(hdr, text="⎘ Copy",
            bg=P["surface"], fg=P["accent2"],
            activebackground=P["border"], activeforeground=P["white"],
            font=("Segoe UI", 7, "bold"), relief="flat",
            padx=6, pady=2, cursor="hand2",
            command=self._copy_steg_output)
        self._steg_copy_btn.pack(side="right")

        # Text box — starts with placeholder text
        self._steg_output = tk.Text(of, height=5, width=28,
            bg=P["surface"], fg=P["muted"],
            insertbackground=P["accent2"],
            font=("Courier New", 9), relief="flat", wrap="word",
            highlightbackground=P["border"], highlightthickness=1,
            state="normal")
        self._steg_output.insert("1.0", "— waiting for decode —")
        self._steg_output.config(state="disabled")
        self._steg_output.pack(fill="x", pady=(0, 4))

        # Character count label
        self._steg_charcount = tk.Label(of, text="", bg=P["card"],
            fg=P["muted"], font=("Segoe UI", 7), anchor="e")
        self._steg_charcount.pack(fill="x")

    # ── TAB: WATERMARK ────────────────────────
    def _build_tab_watermark(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_watermark = p

        self._section(p, "💧 WATERMARK")
        tk.Label(p, text="Stamp text or a logo image onto your photo.",
                 bg=P["card"], fg=P["muted"], font=FONT_SUB,
                 justify="left", wraplength=250).pack(anchor="w", padx=16, pady=(2,8))

        # Mode selector
        self._section(p, "MODE")
        modef = tk.Frame(p, bg=P["card"])
        modef.pack(fill="x", padx=14, pady=(6,0))
        for txt, val in [("Single text (corner)", "single"),
                          ("Tiled diagonal repeat", "tiled"),
                          ("Logo / image stamp",   "stamp")]:
            tk.Radiobutton(modef, text=txt, variable=self._wm_mode, value=val,
                bg=P["card"], fg=P["text"], selectcolor=P["surface"],
                activebackground=P["card"], activeforeground=P["accent"],
                font=("Segoe UI", 8), anchor="w"
            ).pack(fill="x", pady=1)

        # Text input
        self._section(p, "WATERMARK TEXT")
        tf = tk.Frame(p, bg=P["card"])
        tf.pack(fill="x", padx=14, pady=(6,0))
        self._wm_entry = tk.Entry(tf, textvariable=self._wm_text,
            bg=P["surface"], fg=P["white"], insertbackground=P["white"],
            font=("Segoe UI", 9), relief="flat",
            highlightbackground=P["border"], highlightthickness=1)
        self._wm_entry.pack(fill="x", ipady=5)

        # Position
        self._section(p, "POSITION  (single mode)")
        posf = tk.Frame(p, bg=P["card"])
        posf.pack(fill="x", padx=14, pady=(6,0))
        positions = ["top-left","top-right","bottom-left","bottom-right","center"]
        pos_menu = ttk.Combobox(posf, textvariable=self._wm_position,
                                values=positions, state="readonly",
                                font=("Segoe UI", 8))
        pos_menu.pack(fill="x")

        # Opacity
        self._section(p, "OPACITY")
        opf = tk.Frame(p, bg=P["card"])
        opf.pack(fill="x", padx=14, pady=(4,0))
        self._wm_op_disp = tk.Label(opf, text="55%", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI",13,"bold"))
        self._wm_op_disp.pack()
        ttk.Scale(opf, from_=0.05, to=1.0, variable=self._wm_opacity,
                  orient="horizontal",
                  command=lambda v: self._wm_op_disp.config(
                      text=f"{int(float(v)*100)}%")
                  ).pack(fill="x")

        # Logo path (stamp mode)
        self._section(p, "LOGO FILE  (stamp mode)")
        sf = tk.Frame(p, bg=P["card"])
        sf.pack(fill="x", padx=14, pady=(6,0))
        tk.Entry(sf, textvariable=self._wm_stamp_path,
            bg=P["surface"], fg=P["muted"],
            font=("Segoe UI", 7), relief="flat",
            highlightbackground=P["border"], highlightthickness=1
        ).pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(sf, text="…", bg=P["surface"], fg=P["text"],
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            command=self._browse_stamp
        ).pack(side="left", padx=(4,0))

        # Actions
        self._section(p, "ACTIONS", P["accent2"])
        af = tk.Frame(p, bg=P["card"])
        af.pack(fill="x", padx=14, pady=(8,0))
        self._wm_btn = tk.Button(af, text="▶  APPLY WATERMARK",
            bg=P["accent"], fg=P["white"], activebackground="#5a52e0",
            font=FONT_BTN, relief="flat", pady=10, cursor="hand2",
            command=self._run_watermark)
        self._wm_btn.pack(fill="x", pady=(0,6))
        self._reset_btn_for(af)

    # ── TAB: INFO ─────────────────────────────
    def _build_tab_info(self):
        p = tk.Frame(self._tab_frame, bg=P["card"])
        self._panel_info = p
        self._section(p, "ALGORITHM INFO")
        tk.Label(p,
            text=(
                "2D Fast Fourier Transform\n"
                "Cooley-Tukey Radix-2\n\n"
                "Complexity : O(NM log NM)\n"
                "Space      : O(NM)\n\n"
                "Sharpen    : Unsharp mask\n"
                "             + Wiener filter\n\n"
                "Smooth     : Gaussian LP mask\n\n"
                "Filters    : Pillow colour ops\n\n"
                "Blur Tab   : Region censor\n"
                "  Pixelate : Mosaic / news\n"
                "  Gaussian : Smooth fog\n"
                "  Heavy    : Max unrecognisable\n\n"
                "BG Remove  : Corner colour\n"
                "             Luminance thresh\n"
                "             Edge detection\n\n"
                "Steg       : LSB encoding\n"
                "             Hide/reveal msg\n\n"
                "Watermark  : Single / Tiled\n"
                "             Logo stamp\n\n"
                "Channels   : R, G, B separate\n"
                "Save path  : Same as source"
            ),
            bg=P["surface"], fg=P["muted"], font=FONT_MONO,
            justify="left", padx=12, pady=12
        ).pack(fill="x", padx=16, pady=(4, 12))

        self._section(p, "KEYBOARD SHORTCUTS")
        tk.Label(p,
            text="Ctrl+O  Open image\nCtrl+S  Save result\nEnter   Apply current\nEsc     Reset",
            bg=P["surface"], fg=P["muted"], font=FONT_MONO,
            justify="left", padx=12, pady=10
        ).pack(fill="x", padx=16, pady=(4, 16))

    # ── SHARED SIDEBAR HELPERS ────────────────
    def _section(self, parent, label, color=None):
        color = color or P["accent"]
        tk.Frame(parent, bg=P["border"], height=1).pack(fill="x", padx=14, pady=(12, 0))
        tk.Label(parent, text=f"  {label}", bg=P["card"], fg=color,
                 font=("Segoe UI", 7, "bold"), anchor="w"
                 ).pack(fill="x", padx=14, pady=(3, 0))

    def _strength_block(self, parent):
        f = tk.Frame(parent, bg=P["card"])
        f.pack(fill="x", padx=14, pady=(8, 0))
        self.strength_disp = tk.Label(f, text="65%", bg=P["card"],
            fg=P["accent2"], font=("Segoe UI", 20, "bold"))
        self.strength_disp.pack()
        ttk.Scale(f, from_=0.05, to=1.0, variable=self.strength_var,
                  orient="horizontal", command=self._on_strength_change
                  ).pack(fill="x", pady=(2, 0))
        self.strength_desc = tk.Label(f, text="Strong edge enhancement",
            bg=P["card"], fg=P["muted"], font=FONT_SUB, wraplength=240)
        self.strength_desc.pack(pady=(3, 0))

    def _presets_block(self, parent):
        self._section(parent, "PRESETS")
        row = tk.Frame(parent, bg=P["card"])
        row.pack(fill="x", padx=14, pady=(6, 0))
        for i, (name, val) in enumerate([("Subtle",0.25),("Medium",0.55),
                                          ("Strong",0.80),("Max",1.00)]):
            tk.Button(row, text=name, bg=P["surface"], fg=P["text"],
                activebackground=P["accent"], activeforeground=P["white"],
                font=("Segoe UI", 8), relief="flat", padx=4, pady=4,
                cursor="hand2", command=lambda v=val: self._apply_preset(v)
            ).grid(row=0, column=i, padx=2, pady=2, sticky="ew")
            row.columnconfigure(i, weight=1)

    def _reset_btn_for(self, parent):
        tk.Button(parent, text="↺  RESET", bg=P["surface"], fg=P["muted"],
            activebackground=P["border"], activeforeground=P["text"],
            font=("Segoe UI", 9), relief="flat", pady=7, cursor="hand2",
            command=self._reset).pack(fill="x")

    # ── VIEWER ────────────────────────────────────────────────────
    def _build_viewer(self, parent):
        viewer = tk.Frame(parent, bg=P["bg"])
        viewer.pack(side="left", fill="both", expand=True, pady=(0, 12))

        # Header row: labels + zoom sliders side by side per panel
        hdr = tk.Frame(viewer, bg=P["bg"])
        hdr.pack(fill="x", pady=(0, 4))

        self._orig_zoom = tk.DoubleVar(value=1.0)
        self._proc_zoom = tk.DoubleVar(value=1.0)

        for txt, col, zoom_var in [
            ("ORIGINAL IMAGE", P["muted"],   self._orig_zoom),
            ("RESULT IMAGE",   P["accent2"], self._proc_zoom),
        ]:
            h = tk.Frame(hdr, bg=P["bg"])
            h.pack(side="left", fill="x", expand=True, padx=(0, 6))
            tk.Label(h, text=txt, bg=P["bg"], fg=col,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4)
            tk.Frame(h, bg=col, height=2).pack(fill="x", padx=4)
            # Zoom control row
            zrow = tk.Frame(h, bg=P["bg"])
            zrow.pack(fill="x", padx=4, pady=(3, 0))
            tk.Label(zrow, text="🔍 Zoom:", bg=P["bg"], fg=P["muted"],
                     font=("Segoe UI", 7)).pack(side="left")
            tk.Button(zrow, text="−", bg=P["surface"], fg=P["text"],
                      font=("Segoe UI", 8, "bold"), relief="flat",
                      padx=4, pady=0, cursor="hand2",
                      command=lambda z=zoom_var: self._zoom_step(z, -0.15)
                      ).pack(side="left", padx=(4, 1))
            ttk.Scale(zrow, from_=0.2, to=4.0, variable=zoom_var,
                      orient="horizontal",
                      command=lambda v, z=zoom_var: self._on_zoom_change()
                      ).pack(side="left", fill="x", expand=True, padx=2)
            tk.Button(zrow, text="+", bg=P["surface"], fg=P["text"],
                      font=("Segoe UI", 8, "bold"), relief="flat",
                      padx=4, pady=0, cursor="hand2",
                      command=lambda z=zoom_var: self._zoom_step(z, 0.15)
                      ).pack(side="left", padx=(1, 4))
            tk.Button(zrow, text="Fit", bg=P["surface"], fg=P["accent"],
                      font=("Segoe UI", 7, "bold"), relief="flat",
                      padx=5, pady=0, cursor="hand2",
                      command=lambda z=zoom_var: self._zoom_fit(z)
                      ).pack(side="left")

        # Image panels (fill all remaining vertical space)
        panels = tk.Frame(viewer, bg=P["bg"])
        panels.pack(fill="both", expand=True)
        self.orig_panel = self._make_panel(panels, "original")
        self.proc_panel = self._make_panel(panels, "processed")

        self.img_info_var = tk.StringVar(value="No image loaded")
        tk.Label(viewer, textvariable=self.img_info_var, bg=P["bg"],
                 fg=P["muted"], font=FONT_SUB).pack(anchor="w", padx=4, pady=(3, 0))

    def _zoom_step(self, zoom_var, delta):
        zoom_var.set(max(0.2, min(4.0, round(zoom_var.get() + delta, 2))))
        self._on_zoom_change()

    def _zoom_fit(self, zoom_var):
        zoom_var.set(1.0)
        self._on_zoom_change()

    def _on_zoom_change(self):
        self._refresh_both()

    def _make_panel(self, parent, kind):
        outer = tk.Frame(parent, bg=P["card"],
                         highlightbackground=P["border"],
                         highlightthickness=1)
        outer.pack(side="left", fill="both", expand=True,
                   padx=(0, 6) if kind == "original" else 0)

        ph_text = ("📂\n\nOpen an image\nto get started"
                   if kind == "original" else
                   "✦\n\nResult appears\nhere")
        ph = tk.Label(outer, text=ph_text, bg=P["card"], fg=P["border"],
                      font=("Segoe UI", 13))
        ph.pack(expand=True)

        # Scrollable canvas with scrollbars
        canvas_frame = tk.Frame(outer, bg=P["card"])
        canvas = tk.Canvas(canvas_frame, bg="#0d0d14",
                           highlightthickness=0, cursor="fleur")

        hbar = tk.Scrollbar(canvas_frame, orient="horizontal",
                            command=canvas.xview)
        vbar = tk.Scrollbar(canvas_frame, orient="vertical",
                            command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Mouse-wheel scroll
        def _scroll_y(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        def _scroll_x(e):
            canvas.xview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _scroll_y)
        canvas.bind("<Shift-MouseWheel>", _scroll_x)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1,  "units"))

        # Click-drag pan  OR  blur-region draw (depends on active tab)
        def _pan_start(e):
            if kind == "original" and self._active_tab.get() == "blur":
                self._rb_active = True
                self._rb_start  = (e.x, e.y)
                if self._rb_rect_id:
                    canvas.delete(self._rb_rect_id)
                self._rb_rect_id = canvas.create_rectangle(
                    e.x, e.y, e.x, e.y,
                    outline=P["warn"], width=2, dash=(4, 3))
            else:
                canvas.scan_mark(e.x, e.y)

        def _pan_move(e):
            if kind == "original" and self._rb_active:
                if self._rb_rect_id:
                    canvas.delete(self._rb_rect_id)
                self._rb_rect_id = canvas.create_rectangle(
                    self._rb_start[0], self._rb_start[1], e.x, e.y,
                    outline=P["warn"], width=2, dash=(4, 3))
            else:
                canvas.scan_dragto(e.x, e.y, gain=1)

        def _pan_end(e):
            if kind == "original" and self._rb_active:
                self._rb_active = False
                x0, y0 = self._rb_start
                x1, y1 = e.x, e.y
                if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
                    return
                # Convert canvas coords → image pixel coords
                ix = self._canvas_to_image_coords(
                    min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
                if ix is not None:
                    self._blur_rects.append(ix)
                    n = len(self._blur_rects)
                    self._blur_count_var.set(
                        f"{n} region{'s' if n!=1 else ''} drawn  ·  ready to apply")
                    self.status_var.set(
                        f"Region {n} added. Draw more or press ▶ APPLY BLUR REGIONS.")

        canvas.bind("<ButtonPress-1>",   _pan_start)
        canvas.bind("<B1-Motion>",       _pan_move)
        canvas.bind("<ButtonRelease-1>", _pan_end)

        setattr(self, f"{kind}_canvas",        canvas)
        setattr(self, f"{kind}_canvas_frame",  canvas_frame)
        setattr(self, f"{kind}_placeholder",   ph)
        return outer

    # ── DISPLAY ───────────────────────────────────────────────────
    def _show(self, canvas, canvas_frame, panel, placeholder, pil_img, attr, zoom):
        placeholder.pack_forget()
        canvas_frame.pack(fill="both", expand=True)

        pw = max(panel.winfo_width() - 20,  200)
        ph = max(panel.winfo_height() - 20, 200)

        iw, ih = pil_img.size

        # Fit-to-panel base scale, then apply zoom
        base_scale = min(pw / iw, ph / ih)
        scale = base_scale * zoom
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))

        img2   = pil_img.resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img2)
        setattr(self, attr, tk_img)

        canvas.delete("all")
        canvas.config(scrollregion=(0, 0, new_w, new_h))
        # Centre image in viewport when smaller than viewport
        x = max(new_w, pw)  // 2
        y = max(new_h, ph)  // 2
        canvas.create_image(x, y, anchor="center", image=tk_img)

    def _refresh_both(self):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            if self.original_image:
                self._show(self.original_canvas, self.original_canvas_frame,
                           self.orig_panel, self.original_placeholder,
                           self.original_image, "_orig_tk",
                           self._orig_zoom.get())
            if self.processed_image:
                self._show(self.processed_canvas, self.processed_canvas_frame,
                           self.proc_panel, self.processed_placeholder,
                           self.processed_image, "_proc_tk",
                           self._proc_zoom.get())
        finally:
            self._refreshing = False

    # ── EVENTS ────────────────────────────────────────────────────
    def _bind_events(self):
        self.bind("<Configure>", lambda e: self._refresh_both())
        self.bind("<Control-o>", lambda e: self._open_image())
        self.bind("<Control-s>", lambda e: self._save_image())
        self.bind("<Return>",    lambda e: self._on_enter())
        self.bind("<Escape>",    lambda e: self._reset())

    def _on_enter(self):
        t = self._active_tab.get()
        if   t == "sharpen":   self._run_fft("sharpen")
        elif t == "smooth":    self._run_fft("smooth")
        elif t == "filters":   self._run_filter()
        elif t == "blur":      self._run_region_blur()
        elif t == "bgremove":  self._run_bg_remove()
        elif t == "steg":      self._run_steg_hide()
        elif t == "watermark": self._run_watermark()

    def _on_strength_change(self, val=None):
        v = self.strength_var.get()
        try: self.strength_disp.config(text=f"{int(v*100)}%")
        except Exception: pass
        try:
            tab = self._active_tab.get()
            if tab == "smooth":
                descs = ["Subtle smoothing","Moderate blur",
                         "Heavy smoothing","Maximum blur"]
            else:
                descs = ["Gentle edge enhancement","Moderate sharpening",
                         "Strong edge enhancement","Extreme — max clarity"]
            self.strength_desc.config(text=descs[min(int(v/0.25), 3)])
        except Exception:
            pass

    def _apply_preset(self, val):
        self.strength_var.set(val)
        self._on_strength_change()

    # ── REGION BLUR HELPERS ───────────────────────────────────────
    def _canvas_to_image_coords(self, cx0, cy0, cx1, cy1):
        """Convert rubber-band canvas rect to original image pixel rect."""
        if not self.original_image:
            return None, None, None, None
        iw, ih  = self.original_image.size
        panel   = self.orig_panel
        pw      = max(panel.winfo_width()  - 20, 200)
        ph      = max(panel.winfo_height() - 20, 200)
        zoom    = self._orig_zoom.get()
        base    = min(pw / iw, ph / ih)
        scale   = base * zoom
        # Canvas image is centred; top-left offset on canvas
        disp_w, disp_h = int(iw * scale), int(ih * scale)
        ox = max(disp_w, pw)  // 2 - disp_w // 2
        oy = max(disp_h, ph)  // 2 - disp_h // 2
        # Scroll offset
        sx = self.original_canvas.canvasx(0)
        sy = self.original_canvas.canvasy(0)
        # pixel coords
        px0 = int((cx0 + sx - ox) / scale)
        py0 = int((cy0 + sy - oy) / scale)
        px1 = int((cx1 + sx - ox) / scale)
        py1 = int((cy1 + sy - oy) / scale)
        px0 = max(0, min(iw, px0))
        py0 = max(0, min(ih, py0))
        px1 = max(0, min(iw, px1))
        py1 = max(0, min(ih, py1))
        return (px0, py0, px1, py1)

    def _run_region_blur(self):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        if not self._blur_rects:
            messagebox.showwarning("No Regions", "Draw at least one region on the image first.")
            return
        if self._processing:
            return
        self._processing = True
        self._blur_apply_btn.config(text="⏳  Processing…",
                                    state="disabled", bg="#3a3a5a")
        self.status_var.set("Applying blur regions…")
        self.update()
        mode   = self._blur_mode.get()
        rects  = list(self._blur_rects)
        block  = self._blur_block.get()
        radius = self._blur_radius.get()

        def worker():
            try:
                img = self.original_image.copy()
                for rect in rects:
                    x0, y0, x1, y1 = rect
                    if mode == "pixelate":
                        img = RegionBlurEngine.pixelate(img, x0, y0, x1, y1, block)
                    elif mode == "gaussian":
                        img = RegionBlurEngine.gaussian_blur(img, x0, y0, x1, y1, radius)
                    else:
                        img = RegionBlurEngine.heavy_blur(img, x0, y0, x1, y1)
                self.processed_image = img
                self.after(0, lambda: self._show(
                    self.processed_canvas, self.processed_canvas_frame,
                    self.proc_panel, self.processed_placeholder,
                    img, "_proc_tk", self._proc_zoom.get()))
                n = len(rects)
                self.after(0, lambda: self.status_var.set(
                    f"✓ {mode.capitalize()} blur applied to {n} region(s)  |  Ready to save."))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))
            finally:
                self._processing = False
                self.after(0, lambda: self._blur_apply_btn.config(
                    text="▶  APPLY BLUR REGIONS", state="normal", bg=P["accent"]))

        threading.Thread(target=worker, daemon=True).start()

    def _clear_blur_rects(self):
        self._blur_rects = []
        self._blur_count_var.set("No regions drawn yet")
        # Remove rubber-band rect from canvas if still visible
        if self._rb_rect_id:
            try:
                self.original_canvas.delete(self._rb_rect_id)
            except Exception:
                pass
            self._rb_rect_id = None
        self.status_var.set("All blur regions cleared.")

    # ── BACKGROUND REMOVAL ACTIONS ────────────────────────────────
    def _run_bg_remove(self):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        if self._processing:
            return
        self._processing = True
        self._bgr_btn.config(text="⏳  Processing…", state="disabled", bg="#1a2a4a")
        self.status_var.set("Removing background…")
        self.update()
        method    = self._bg_method.get()
        tolerance = self._bg_tolerance.get()
        threshold = self._bg_threshold.get()
        invert    = self._bg_invert_var.get()
        hue       = self._bg_hue_var.get()
        bg_col    = (self._bg_color_r.get(),
                     self._bg_color_g.get(),
                     self._bg_color_b.get())

        def worker():
            try:
                if method == "color":
                    rgba = BackgroundRemovalEngine.by_color(
                        self.original_image, tolerance)
                elif method == "threshold":
                    rgba = BackgroundRemovalEngine.by_threshold(
                        self.original_image, threshold, invert)
                elif method == "edge":
                    rgba = BackgroundRemovalEngine.by_edge(self.original_image)
                else:  # chroma
                    rgba = BackgroundRemovalEngine.by_chroma(
                        self.original_image, hue_center=hue)

                result = BackgroundRemovalEngine.replace_background(rgba, bg_col)
                self.processed_image = result
                self.after(0, lambda: self._show(
                    self.processed_canvas, self.processed_canvas_frame,
                    self.proc_panel, self.processed_placeholder,
                    result, "_proc_tk", self._proc_zoom.get()))
                self.after(0, lambda: self.status_var.set(
                    f"✓ Background removed ({method} method)  |  Ready to save."))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))
            finally:
                self._processing = False
                self.after(0, lambda: self._bgr_btn.config(
                    text="▶  REMOVE BACKGROUND", state="normal", bg="#2a4a7a"))

        threading.Thread(target=worker, daemon=True).start()

    # ── STEGANOGRAPHY ACTIONS ─────────────────────────────────────
    def _run_steg_hide(self):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        msg = self._steg_entry.get("1.0", "end").strip()
        if not msg:
            messagebox.showwarning("No Message", "Type a secret message first.")
            return
        if self._processing:
            return
        # Show capacity check immediately
        cap = SteganographyEngine.capacity(self.original_image)
        needed = len(msg.encode("utf-8"))
        if needed > cap:
            messagebox.showerror("Message Too Long",
                f"Message needs {needed} bytes but image can only carry {cap} bytes.\n"
                f"Use a larger image or a shorter message.")
            return
        self._processing = True
        self._steg_hide_btn.config(text="⏳  Encoding…", state="disabled", bg="#3a3a5a")
        self.status_var.set(f"Hiding message ({needed} bytes into {cap} available)…")

        def worker():
            try:
                result = SteganographyEngine.hide(self.original_image, msg)
                self.processed_image = result
                self.after(0, lambda: self._show(
                    self.processed_canvas, self.processed_canvas_frame,
                    self.proc_panel, self.processed_placeholder,
                    result, "_proc_tk", self._proc_zoom.get()))
                self.after(0, lambda: self.status_var.set(
                    f"✓ Message hidden ({needed} bytes)  |  Save as PNG — NOT JPEG!"))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))
            finally:
                self._processing = False
                self.after(0, lambda: self._steg_hide_btn.config(
                    text="🔒  HIDE MESSAGE", state="normal", bg=P["accent"]))

        threading.Thread(target=worker, daemon=True).start()

    def _run_steg_reveal(self):
        # Decide which image to decode from:
        # — If the user just hid a message, the encoded image is in processed_image
        # — If the user opened a saved carrier PNG, it is in original_image
        img_to_decode = self.processed_image or self.original_image
        if img_to_decode is None:
            messagebox.showwarning("No Image",
                "Open an image (or hide a message first) before revealing.")
            return

        source_label = "processed image" if self.processed_image else "original image"

        self._steg_reveal_btn.config(text="⏳  Decoding…", state="disabled",
                                     bg="#1a3a2a")
        self.status_var.set(f"Decoding hidden message from {source_label}…")
        self.update()

        def worker():
            try:
                msg = SteganographyEngine.reveal(img_to_decode)
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Decode Error", str(ex)))
                self.after(0, lambda: self._steg_reveal_btn.config(
                    text="🔍  REVEAL MESSAGE", state="normal", bg="#2a5a4a"))
                return

            if msg and msg.strip():
                display   = msg
                colour    = P["accent2"]
                count_txt = f"{len(msg)} character{'s' if len(msg)!=1 else ''} decoded"
                status    = f"✓ Message decoded from {source_label} — {len(msg)} chars"
            else:
                display   = "(no hidden message found in this image)"
                colour    = P["warn"]
                count_txt = ""
                status    = f"⚠ No hidden message found in {source_label}"

            def show():
                w = self._steg_output
                w.config(state="normal", fg=colour)
                w.delete("1.0", "end")
                w.insert("end", display)
                w.config(state="disabled")
                self._steg_charcount.config(text=count_txt)
                self.status_var.set(status)
                self._steg_reveal_btn.config(
                    text="🔍  REVEAL MESSAGE", state="normal", bg="#2a5a4a")
                self.after(150, lambda: self._side_canvas.yview_moveto(1.0))

            self.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    def _copy_steg_output(self):
        """Copy the decoded message text to the system clipboard."""
        try:
            self._steg_output.config(state="normal")
            text = self._steg_output.get("1.0", "end").strip()
            self._steg_output.config(state="disabled")
            if text and text != "— waiting for decode —" \
                    and not text.startswith("(no hidden"):
                self.clipboard_clear()
                self.clipboard_append(text)
                self._steg_copy_btn.config(text="✓ Copied!", fg=P["accent2"])
                self.after(1500, lambda: self._steg_copy_btn.config(
                    text="⎘ Copy", fg=P["accent2"]))
                self.status_var.set("Message copied to clipboard.")
            else:
                self.status_var.set("Nothing to copy — decode a message first.")
        except Exception as ex:
            messagebox.showerror("Copy Error", str(ex))

    # ── WATERMARK ACTIONS ─────────────────────────────────────────
    def _browse_stamp(self):
        path = filedialog.askopenfilename(
            title="Select logo/stamp image",
            filetypes=[("Image files","*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All files","*.*")])
        if path:
            self._wm_stamp_path.set(path)

    def _run_watermark(self):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        if self._processing:
            return
        mode     = self._wm_mode.get()
        text     = self._wm_text.get().strip()
        position = self._wm_position.get()
        opacity  = self._wm_opacity.get()
        stamp_p  = self._wm_stamp_path.get().strip()

        if mode == "stamp" and not stamp_p:
            messagebox.showwarning("No Logo", "Please browse and select a logo file first.")
            return
        if mode != "stamp" and not text:
            messagebox.showwarning("No Text", "Please enter watermark text first.")
            return

        self._processing = True
        self._wm_btn.config(text="⏳  Stamping…", state="disabled", bg="#3a3a5a")
        self.status_var.set("Applying watermark…")

        def worker():
            try:
                if mode == "single":
                    result = WatermarkEngine.text(
                        self.original_image, text, position, opacity)
                elif mode == "tiled":
                    result = WatermarkEngine.tiled(
                        self.original_image, text, opacity)
                else:
                    result = WatermarkEngine.image_stamp(
                        self.original_image, stamp_p, position, opacity)
                self.processed_image = result
                self.after(0, lambda: self._show(
                    self.processed_canvas, self.processed_canvas_frame,
                    self.proc_panel, self.processed_placeholder,
                    result, "_proc_tk", self._proc_zoom.get()))
                self.after(0, lambda: self.status_var.set(
                    f"✓ Watermark applied ({mode} mode)  |  Ready to save."))
            except Exception as ex:
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))
            finally:
                self._processing = False
                self.after(0, lambda: self._wm_btn.config(
                    text="▶  APPLY WATERMARK", state="normal", bg=P["accent"]))

        threading.Thread(target=worker, daemon=True).start()

    # ── CORE ACTIONS ──────────────────────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files","*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All files","*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.original_image  = img
            self.processed_image = None
            self.image_path      = path
            try: self.processed_canvas_frame.pack_forget()
            except Exception: pass
            self.processed_placeholder.pack(expand=True)
            self._show(self.original_canvas, self.original_canvas_frame,
                       self.orig_panel, self.original_placeholder,
                       img, "_orig_tk", self._orig_zoom.get())
            w, h = img.size
            fname = os.path.basename(path)
            self.img_info_var.set(
                f"File: {fname}  |  {w} × {h} px  |  Pixels: {w*h:,}"
                f"  |  FFT ≈ {w*h*int(np.log2(max(w*h,2))):,} ops")
            self.status_var.set(f"Loaded: {fname} ({w}×{h})")
            self._populate_filter_thumbs()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")

    def _run_fft(self, mode):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        if self._processing:
            return
        self._processing = True
        btn = self._sharpen_btn if mode == "sharpen" else self._smooth_btn
        btn.config(text="⏳  Processing…", state="disabled", bg="#3a3a5a")
        self.status_var.set("Applying FFT filter…")
        self.update()

        def worker():
            try:
                result = FourierEngine.process(self.original_image, mode,
                                               self.strength_var.get())
                self.processed_image = result
                self.after(0, lambda: self._show(
                    self.processed_canvas, self.processed_canvas_frame,
                    self.proc_panel, self.processed_placeholder,
                    result, "_proc_tk", self._proc_zoom.get()))
                label = "Sharpened" if mode == "sharpen" else "Smoothed"
                w, h  = result.size
                self.after(0, lambda: self.status_var.set(
                    f"✓ {label} at {int(self.strength_var.get()*100)}%"
                    f"  |  {w}×{h}  |  Ready to save."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self._processing = False
                lbl = "▶  SHARPEN IMAGE" if mode == "sharpen" else "▶  SMOOTH IMAGE"
                self.after(0, lambda: btn.config(text=lbl, state="normal",
                                                  bg=P["accent"]))

        threading.Thread(target=worker, daemon=True).start()

    def _run_filter(self):
        if not self.original_image:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        if self._processing:
            return
        idx = self._active_filter.get()
        self._processing = True
        self.filter_apply_btn.config(text="⏳  Applying…",
                                     state="disabled", bg="#3a3a5a")
        name = FilterEngine.FILTERS[idx][0]
        self.status_var.set(f"Applying filter: {name}…")
        self.update()

        def worker():
            try:
                result = FilterEngine.apply(self.original_image, idx)
                self.processed_image = result
                self.after(0, lambda: self._show(
                    self.processed_canvas, self.processed_canvas_frame,
                    self.proc_panel, self.processed_placeholder,
                    result, "_proc_tk", self._proc_zoom.get()))
                w, h = result.size
                self.after(0, lambda: self.status_var.set(
                    f"✓ Filter applied: {name}  |  {w}×{h}  |  Ready to save."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self._processing = False
                self.after(0, lambda: self.filter_apply_btn.config(
                    text="▶  APPLY FILTER", state="normal", bg=P["accent"]))

        threading.Thread(target=worker, daemon=True).start()

    def _save_image(self):
        if not self.processed_image:
            messagebox.showwarning("Nothing to Save", "Process an image first.")
            return
        # Default save location = same folder as source image
        init_dir  = os.path.dirname(self.image_path) if self.image_path else os.getcwd()
        init_name = ""
        if self.image_path:
            base, ext = os.path.splitext(os.path.basename(self.image_path))
            tag_map = {
                "sharpen":   "sharpened",
                "smooth":    "smoothed",
                "filters":   FilterEngine.FILTERS[
                    self._active_filter.get()][0].lower().replace(" ", "_"),
                "blur":      "censored",
                "bgremove":  "bg_removed",
                "steg":      "hidden_msg",
                "watermark": "watermarked",
                "info":      "result",
            }
            tag       = tag_map.get(self._active_tab.get(), "result")
            init_name = f"{base}_{tag}{ext or '.png'}"

        path = filedialog.asksaveasfilename(
            title="Save Result",
            initialdir=init_dir,
            initialfile=init_name,
            defaultextension=".png",
            filetypes=[("PNG (lossless)", "*.png"),
                       ("JPEG",           "*.jpg"),
                       ("TIFF",           "*.tiff"),
                       ("BMP",            "*.bmp")])
        if not path:
            return
        try:
            self.processed_image.save(path)
            self.status_var.set(f"✓ Saved → {path}")
            messagebox.showinfo("Saved", f"Image saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _reset(self):
        self.processed_image = None
        if not self.original_image:
            return
        try: self.processed_canvas_frame.pack_forget()
        except Exception: pass
        self.processed_placeholder.pack(expand=True)
        self.status_var.set("Reset — ready to process.")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = FourierImageStudio()
    app.update_idletasks()
    w, h = 1340, 860
    sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
    app.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    app.mainloop()
