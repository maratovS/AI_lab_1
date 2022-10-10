import tensorflow as tf
from tensorflow import keras
import numpy as np


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

model = keras.models.load_model('my_model')

daisy_path = "daisy.jpg"
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
img_height = 180
img_width = 180


img = tf.keras.utils.load_img(
    daisy_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# class_name = keras.np_utils.probas_to_classes(predictions)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)