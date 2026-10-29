"""Render a word to a fixed-size canvas with per-letter bounding boxes.

Conventions:
  - Images are float32 in [0, 1] with ink = 1 and background = 0, so an
    all-zero image is a blank page (a meaningful attribution baseline).
  - Text is drawn left-aligned at a fixed origin (pen x, baseline y), plus an
    integer jitter (dx, dy). The canvas is sized so every vocab word in every
    font fits for any jitter in range.
  - Boxes are (x0, y0, x1, y1) in pixels, half-open: pixels x0 <= x < x1.
    Two kinds are returned per letter:
      ink:  the glyph's tight ink box (may overlap neighbors or leave gaps)
      cell: [pen position of this letter, pen position of the next) x
            [baseline - ascent, baseline + descent]; cells tile the word.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "fonts"
FONT_NAMES = sorted(p.stem for p in FONT_DIR.glob("*.ttf"))

FONT_PX = 16       # pixels per em
WEIGHT = 400       # variable fonts default to different weights (200-400); fix it
MARGIN = 2         # blank pixels between the jitter range and the canvas edge
JITTER_X = 4       # dx in [-JITTER_X, JITTER_X]
JITTER_Y = 2       # dy in [-JITTER_Y, JITTER_Y]


def load_font(name, size=FONT_PX):
    font = ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size)
    axes = font.get_variation_axes()
    font.set_variation_by_axes([WEIGHT if a["name"] in (b"Weight", "Weight") else a["default"]
                                for a in axes])
    return font


class Canvas:
    """Fixed canvas geometry shared by all fonts, computed from the vocabulary."""

    def __init__(self, fonts, vocab):
        left, top, right, bottom = 0, 0, 0, 0
        for font in fonts.values():
            for w in vocab:
                l, t, r, b = font.getbbox(w, anchor="ls")   # relative to baseline-left
                left, top = min(left, l), min(top, t)
                right, bottom = max(right, r), max(bottom, b)
        pad_x, pad_y = MARGIN + JITTER_X, MARGIN + JITTER_Y
        self.pen_x = pad_x - left                 # pen start x at dx = 0
        self.baseline = pad_y - top               # baseline y at dy = 0
        self.width = self.pen_x + math.ceil(right) + pad_x
        self.height = self.baseline + math.ceil(bottom) + pad_y


def letter_origins(font, word):
    """Pen x of each letter relative to the word origin.

    getlength(word[:i+1]) includes any kerning between letters i-1 and i;
    subtracting letter i's own advance gives where letter i starts.
    """
    return [font.getlength(word[:i + 1]) - font.getlength(word[i]) for i in range(len(word))]


def letter_boxes(font, word, x, y):
    """Ink and cell boxes for each letter when the word is drawn at pen (x, baseline y)."""
    origins = letter_origins(font, word)
    ascent, descent = font.getmetrics()
    ink, cell = [], []
    for i, c in enumerate(word):
        ox = x + origins[i]
        l, t, r, b = font.getbbox(c, anchor="ls")
        ink.append((math.floor(ox + l), y + t, math.ceil(ox + r), y + b))
        nx = x + (origins[i + 1] if i + 1 < len(word) else font.getlength(word))
        cell.append((round(ox), y - ascent, round(nx), y + descent))
    return ink, cell


def draw(font, text, canvas, x, y):
    img = Image.new("L", (canvas.width, canvas.height), 0)
    ImageDraw.Draw(img).text((x, y), text, fill=255, font=font, anchor="ls")
    return np.asarray(img, dtype=np.float32) / 255.0


def render(word, font, canvas, dx=0, dy=0):
    """Returns (image HxW float32, ink boxes, cell boxes)."""
    assert abs(dx) <= JITTER_X and abs(dy) <= JITTER_Y
    x, y = canvas.pen_x + dx, canvas.baseline + dy
    ink, cell = letter_boxes(font, word, x, y)
    return draw(font, word, canvas, x, y), ink, cell
