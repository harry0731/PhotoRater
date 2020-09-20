from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

img = Image.open('/home/hugo/Downloads/photo-1481930079977-24a345fcae85.jpeg')
img = img.resize((224,224), Image.ANTIALIAS)
img.save('RDs0THr4lGs.jpg') 