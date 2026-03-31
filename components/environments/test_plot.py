from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create a test image
rgb = np.zeros((300, 300, 3), dtype=np.uint8)
vis_img = Image.fromarray(rgb)
draw = ImageDraw.Draw(vis_img)
draw.rectangle([10, 10, 50, 50], outline="green", width=2)
vis_img.save("test.png")
