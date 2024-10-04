from PIL import Image

# Open and re-save the image
with Image.open('desert.JPEG') as img:
    img.convert("RGB").save('style_standardized_image.jpg', format='JPEG')
