import sys, glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
files = sys.argv[2:]; out = sys.argv[1]
ims = []
for f in files:
    im = Image.open(f); s = 640 / im.width if im.width < 4000 else 1300 / im.width
    ims.append(im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS))
W = 1320; x = y = rowh = 0; pos = []
for im in ims:
    if x + im.width > W: x = 0; y += rowh + 20; rowh = 0
    pos.append((x, y)); x += im.width + 20; rowh = max(rowh, im.height)
sheet = Image.new("RGB", (W, y + rowh), "white")
for im, p in zip(ims, pos): sheet.paste(im.convert("RGB"), p)
sheet.save(out)
