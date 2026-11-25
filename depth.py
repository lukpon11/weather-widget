# Erzeugt eine Tiefenkarte (hell = nah) für ein Bild mit Transparenz; außerhalb der Maske schwarz.
import sys
from PIL import Image, ImageFilter
import numpy as np
from transformers import pipeline

src, out = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGBA")
rgb = Image.new("RGB", img.size, (255, 255, 255)); rgb.paste(img, mask=img.split()[3])
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
d = np.asarray(pipe(rgb)["depth"].resize(img.size, Image.BICUBIC), dtype=np.float32)
mask = np.asarray(img.split()[3]) > 128
lo, hi = np.percentile(d[mask], 1), np.percentile(d[mask], 99)
d = np.clip((d - lo) / (hi - lo), 0, 1); d[~mask] = 0
Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)).save(out)
print("ok", img.size, "mask px", int(mask.sum()))
