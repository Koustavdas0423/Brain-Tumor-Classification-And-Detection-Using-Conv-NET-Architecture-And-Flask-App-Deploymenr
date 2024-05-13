import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('C:/Users/koust/Downloads/Brain-Tumor-Detection-master/models/cnn-parameters-improvement-23-0.91.model')

def get_class_name(value):
    image = cv2.imread(value)
    img = Image.fromarray(image)
    img = img.resize((240, 240))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    result = model.predict(input_img)
    tumor_probability = result[0]  # Probability of tumor class
    threshold = 0.5
    if tumor_probability >= threshold:
        return "Tumor detected."
    else:
        return "No tumor detected."


# Example usage
result = get_class_name('C:/Users/koust/Downloads/Brain-Tumor-Detection-master/augmented data/no/aug_1 no._0_237.jpg')
print(result)
