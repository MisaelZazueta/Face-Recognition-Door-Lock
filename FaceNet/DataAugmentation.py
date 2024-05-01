from numpy import expand_dims
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
import cv2
import os

print("Nombre: ")
name = input()
# name = "Misa"

directory = name
parent_dir = r"C:\Users\ingmi\PycharmProjects\FaceNet-rPPG\FaceNet\dataset"
path = os.path.join(parent_dir, directory)
os.mkdir(path)
dir = parent_dir + "\\" + name
img_n = name + ".png"

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = img_n.format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        break
cam.release()
cv2.destroyAllWindows()

# load the image
img = load_img(img_n)
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=1, save_to_dir=dir)

# create image data augmentation generator
for i in range(1):
    batch = it.next()
    image = batch[0].astype('uint8')

datagen = ImageDataGenerator(horizontal_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1, save_to_dir=dir)

# create image data augmentation generator
for i in range(1):
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
datagen = ImageDataGenerator(shear_range=30)
# prepare iterator
it = datagen.flow(samples, batch_size=1, save_to_dir=dir)

# generate samples and plot
for i in range(1):
 # generate batch of images
 batch = it.next()
 # convert to unsigned integers for viewing
 image = batch[0].astype('uint8')
os.remove(img_n)
