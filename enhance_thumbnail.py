from PIL import ImageFont, ImageDraw, Image

def draw_thumbnail_text(img_path, text, out_path):
    img = Image.open(img_path).convert('RGBA')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    margin = 60
    max_width = w - 2 * margin
    font_size = 100
    min_font_size = 80
    font = None
    # Robust font fallback
    try:
        font = ImageFont.truetype('/Library/Fonts/BebasNeue-Bold.ttf', font_size)
    except:
        try:
            font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', font_size)
        except:
            font = ImageFont.load_default()
            font_size = 40

    # Try to fit text in one line, truncate if needed
    max_chars = 32  # You can adjust this for your thumbnail width
    if len(text) > max_chars:
        text = text[:max_chars-1] + '…'

    # Reduce font size if text is too wide, but never below min_font_size
    while True:
        bbox = draw.textbbox((0,0), text, font=font)
        text_w = bbox[2] - bbox[0]
        if text_w <= max_width or font_size <= min_font_size:
            break
        font_size -= 4
        try:
            font = ImageFont.truetype('/Library/Fonts/BebasNeue-Bold.ttf', font_size)
        except:
            try:
                font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', font_size)
            except:
                font = ImageFont.load_default()
                font_size = 40

    print(f"[THUMBNAIL] Final font size used: {font_size}")

    # Always use one line for max size
    text_lines = [text]

    # Calculate total text height
    total_text_h = draw.textbbox((0,0), text, font=font)[3] - draw.textbbox((0,0), text, font=font)[1]
    y = h - total_text_h - 60

    # Draw bar
    bar_height = total_text_h + 40
    bar = Image.new('RGBA', (w, int(bar_height)), (0,0,0,180))
    img.alpha_composite(bar, (0, int(y-20)))

    # Draw text with bold outline and gold color
    bbox = draw.textbbox((0,0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = (w - text_w) // 2
    for dx in [-4, 0, 4]:
        for dy in [-4, 0, 4]:
            if dx != 0 or dy != 0:
                draw.text((int(x+dx), int(y+dy)), text, font=font, fill='black')
    draw.text((int(x), int(y)), text, font=font, fill=(255, 215, 0))

    img.convert('RGB').save(out_path)
    print(f'Enhanced image with adaptive text saved to: {out_path}')

def add_visual_hook(img_path, out_path, emoji='😲', position='top-right'):
    img = Image.open(img_path).convert('RGBA')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    emoji_font_size = 180
    margin = 40
    # Try to use Apple Color Emoji font for best emoji rendering
    font_paths = [
        '/System/Library/Fonts/Apple Color Emoji.ttc',  # macOS
        '/Library/Fonts/Apple Color Emoji.ttf',         # alternate macOS
        '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf', # Linux fallback
        '/Library/Fonts/Arial Unicode.ttf',             # fallback
    ]
    emoji_font = None
    for fp in font_paths:
        try:
            emoji_font = ImageFont.truetype(fp, emoji_font_size)
            break
        except:
            continue
    if emoji_font is None:
        emoji_font = ImageFont.load_default()
    # Calculate position (top-right, with margin)
    x, y = w - emoji_font_size - margin, margin
    # Draw white ellipse background for contrast
    ellipse_bbox = [x-20, y-20, x+emoji_font_size+20, y+emoji_font_size+20]
    draw.ellipse(ellipse_bbox, fill=(255,255,255,230))
    # Draw subtle shadow (black, slightly offset)
    shadow_offset = 6
    draw.text((x+shadow_offset, y+shadow_offset), emoji, font=emoji_font, fill=(0,0,0,120))
    # Draw emoji
    draw.text((x, y), emoji, font=emoji_font, fill=None)
    img.convert('RGB').save(out_path)
    print(f'Visual hook added and saved to: {out_path}')

if __name__ == '__main__':
    long_title = "King Bhoj's Surprising Secret to Everlasting Happiness Revealed!"
    draw_thumbnail_text('thumbnails/thumbnail_test.jpg', long_title, 'thumbnails/thumbnail_test_adaptive_text.jpg')
    add_visual_hook('thumbnails/thumbnail_test_adaptive_text.jpg', 'thumbnails/thumbnail_test_adaptive_text_hook.jpg', emoji='😲', position='top-right') 