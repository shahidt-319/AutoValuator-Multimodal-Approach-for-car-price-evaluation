
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model('image_model/damage_detector.h5', compile=False)

class_names = ['minor', 'moderate', 'severe']

# Penalty values
penalty_map = {'minor': 0.1, 'moderate': 0.2, 'severe': 0.3}

def predict_damage(image_path):
    img = image.load_img(image_path, target_size=(80, 80))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class_index = np.argmax(pred)
    damage_class = class_names[pred_class_index]
    damage_penalty = penalty_map[damage_class]

    return damage_class, damage_penalty
