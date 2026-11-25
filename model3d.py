# Erzeugt aus der 2D-Illustration die Daten für das 3D-Modell (Tab "3D-Ansicht"):
# 3d/ground.png (Bodentextur ohne gezeichnete Bäume/Schatten), 3d/height.png, 3d/water.png, 3d/hole.json
# Aufruf: uv run --with numpy --with pillow --with scipy --with scikit-image python model3d.py <illustration.png> 3d
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import find_contours, approximate_polygon

src, out = sys.argv[1], sys.argv[2]
im = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
H, W = im.shape[:2]
rgb, a = im[..., :3] / 255, im[..., 3]
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
mx, mn = rgb.max(-1), rgb.min(-1)
v, d = mx, np.maximum(mx - mn, 1e-6)
s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60

# Flächen nach Farbe (Schwellen an dieser Illustration abgelesen)
inside = a > 128
water = inside & (h > 165) & (h < 220) & (s > 0.2)
bunker = inside & (v > 0.82) & (s < 0.3)
path = inside & ~bunker & (s < 0.2) & (v > 0.35)
hut = inside & ((h < 40) | (h > 300)) & (s > 0.3) & (v < 0.75) & (v > 0.12)
greenish = inside & ~water & ~bunker & ~path & ~hut
mowed = greenish & (v >= 0.52)
dark = greenish & (v < 0.30)                       # gezeichnete Bäume + Schatten
lab, n = ndi.label(ndi.binary_opening(mowed, iterations=2))
sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
fair_id = int(np.argmax(sizes)) + 1
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
green = (lab == fair_id) & (xx < 255) & (yy < 230)  # Grün hängt oben links am Fairway
fairway = (lab == fair_id) & ~green
tees = np.zeros_like(inside)
for i in range(1, n + 1):
    cy, cx = ndi.center_of_mass(lab == i)
    if i != fair_id and sizes[i - 1] > 1000 and cy > 650: tees |= lab == i

# Bodentextur: Original, gezeichnete Bäume/Schatten durch Rough-Farbe mit leichtem Rauschen ersetzt
rough_px = greenish & ~mowed & ~dark
rough_col = np.median(im[rough_px][:, :3], axis=0)
tex = im.copy()
noise = ndi.gaussian_filter(np.random.default_rng(1).normal(0, 1, (H, W)), 2) * 18
fill = (ndi.binary_dilation(dark, iterations=3) & ~(mowed | path | bunker | water)) | ndi.binary_dilation(hut, iterations=4)  # Hütte kommt als 3D-Objekt
for c in range(3): tex[..., c] = np.where(fill, np.clip(rough_col[c] + noise, 0, 255), tex[..., c])
tex[..., 3] = np.where(inside, 255, 0)
Image.fromarray(tex.astype(np.uint8)).save(f"{out}/ground.png")

# Höhenkarte (0.5 = Normalniveau): sanfte Hügel, Abschläge/Grün erhöht, Bunker/Wasser abgesenkt
hm = 0.5 + ndi.gaussian_filter(np.random.default_rng(2).normal(0, 1, (H, W)), 40) * 3.0
hm = np.where(tees, 0.62, hm)
hm = np.where(green, hm + 0.04, hm)
hm = np.where(ndi.binary_dilation(bunker, iterations=3), 0.36, hm)
hm = np.where(ndi.binary_dilation(water, iterations=2), 0.22, hm)
hm = ndi.gaussian_filter(hm, 3)
Image.fromarray((np.clip(hm, 0, 1) * 255).astype(np.uint8)).save(f"{out}/height.png")
Image.fromarray((ndi.binary_dilation(water, iterations=3) * 255).astype(np.uint8)).save(f"{out}/water.png")

# Bäume: Mittelpunkte in den dunklen Flächen, größte Kronen zuerst, ohne starke Überlappung
dt = ndi.distance_transform_edt(ndi.binary_opening(dark, iterations=1))
ys, xs = np.nonzero(dt >= 3)
order = np.argsort(-dt[ys, xs])
trees = []
for k in order:
    y, x = ys[k], xs[k]; rr = float(min(max(dt[y, x] * 1.1, 5), 22))
    if all((x - tx) ** 2 + (y - ty) ** 2 > (0.62 * (rr + tr)) ** 2 for tx, ty, tr in trees):
        trees.append((int(x), int(y), rr))

hl, hn = ndi.label(hut)
hut_c = ndi.center_of_mass(hl == (np.argmax(ndi.sum(hut, hl, range(1, hn + 1))) + 1))
green_c = ndi.center_of_mass(green)
outline = max(find_contours(np.pad(inside, 1).astype(float), 0.5), key=len)
outline = approximate_polygon(outline - 1, tolerance=1.5)

json.dump({
    "w": W, "h": H,
    "outline": [[round(p[1] / W, 4), round(p[0] / H, 4)] for p in outline],
    "trees": [[round(x / W, 4), round(y / H, 4), round(rr / W, 4)] for x, y, rr in trees],
    "hut": [round(hut_c[1] / W, 4), round(hut_c[0] / H, 4)],
    "flag": [round(green_c[1] / W, 4), round(green_c[0] / H, 4)],
}, open(f"{out}/hole.json", "w"))
print("trees", len(trees), "outline pts", len(outline), "hut", hut_c, "green", green_c)
