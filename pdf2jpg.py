import os
from pdf2image import convert_from_path

# filenameslist of name of files or images in data set folder

pdfs_path = "./static/pdf/"
img_path = "./static/img/"

filenames = list()

for image in os.walk("./static/pdf"):
    filenames.append(image[2]) 

for img in filenames[0]:
  name = pdfs_path + img
  images = convert_from_path(name, dpi=200)
  for img in images:
  #Get the file name
    name = name.split('/')[-1]
    for idx, im in enumerate(images):
		  #Change the file name
      name = name.replace('.pdf', '.jpg')
		  #Remove white spaces
      name = name.replace(' ', '_')
      name = name.replace('\\ ', '_')
      #Define the name of the image
      file_name = img_path + name
      #Save the image
      img.save(file_name, 'JPEG')
