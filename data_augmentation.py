
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from distutils.dir_util import copy_tree

# Filenameslist of name of files or images in data set folder 
filenames = list()

img_path = 'static/img/'
aug_path = 'static/aug/'

# Dataaugmentation

for image in os.walk(img_path):
    filenames.append(image[2]) 

datagen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               zca_whitening=False,
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])

for f in filenames[0]:
    img = load_img(img_path+f)
    x = img_to_array(img) 
    # Reshape the input image 
    x = x.reshape((1, ) + x.shape)  
    i = 0

    # generate 5 new augmented images 
    for batch in datagen.flow(x, batch_size=1,
                      save_to_dir =aug_path,
                      save_prefix ='aug', save_format='jpg'):
        i += 1
        if i > 5: 
            break
            
# Copy from image_path to image_path

copy_tree(img_path, aug_path)
