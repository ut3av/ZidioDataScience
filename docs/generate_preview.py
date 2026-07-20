from PIL import Image, ImageDraw, ImageFont

img = Image.new('RGB', (1400, 800), color='#f8fbff')
draw = ImageDraw.Draw(img)
draw.rounded_rectangle([60, 60, 1340, 740], radius=32, fill='#ffffff', outline='#dbeafe')
draw.rounded_rectangle([90, 100, 1310, 220], radius=26, fill='#eff6ff', outline='#dbeafe')
draw.rounded_rectangle([90, 260, 1310, 700], radius=26, fill='#ffffff', outline='#dbeafe')

try:
    font_title = ImageFont.truetype('arial.ttf', 46)
    font_sub = ImageFont.truetype('arial.ttf', 24)
    font_small = ImageFont.truetype('arial.ttf', 18)
    font_label = ImageFont.truetype('arial.ttf', 20)
except Exception:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_label = ImageFont.load_default()

draw.text((120, 120), 'NeuralRetail Dashboard', fill='#0f172a', font=font_title)
draw.text((120, 182), 'AI-powered retail analytics for beginners', fill='#2563eb', font=font_sub)

for i, label in enumerate(['Revenue', 'Customers', 'Forecast', 'Inventory']):
    x = 120 + i * 280
    draw.rounded_rectangle([x, 300, x + 220, 390], radius=20, fill='#ffffff', outline='#dbeafe')
    draw.text((x + 20, 320), label, fill='#64748b', font=font_small)
    draw.text((x + 20, 350), '$1.2M', fill='#111827', font=font_sub)

draw.text((120, 440), 'Open the app and explore filters, KPIs, forecasting, and inventory insights', fill='#334155', font=font_label)
img.save('docs/neuralretail_preview.png')
print('created docs/neuralretail_preview.png')
