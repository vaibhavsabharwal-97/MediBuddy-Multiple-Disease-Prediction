import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread("D:\\custom\\project\\uploads\\Y1.jpg")

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

predictions = model.predict(input_img)
predicted_class_index = np.argmax(predictions, axis=1)
print("Predicted class index:", predicted_class_index)
