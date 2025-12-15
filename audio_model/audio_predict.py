import os
import json
import numpy as np
import librosa
import tensorflow as tf
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_cnn_gru_model.h5")
LABEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_gru_labels.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    label_to_index = json.load(f)
index_to_label = [None] * len(label_to_index)
for label, idx in label_to_index.items():
    index_to_label[idx] = label

def pad_mfcc(mfcc, max_frames=100):
    if mfcc.shape[0] > max_frames:
        return mfcc[:max_frames, :]
    else:
        pad_amt = max_frames - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad_amt), (0,0)), mode='constant')

def predict_engine_sound_cnn_gru(preprocessed_audio_path, min_samples=22050):
    notes = []
    try:
        y, sr = librosa.load(preprocessed_audio_path, sr=22050)
        notes.append(f"Loaded audio, duration: {y.shape[0]/sr:.2f}s")
        if len(y) < min_samples: 
            msg = f"Warning: Cleaned audio is very short ({len(y)} samples). Prediction may be unreliable."
            notes.append(msg)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T
        mfcc_padded = pad_mfcc(mfcc, 100)
        input_array = np.expand_dims(mfcc_padded, axis=0)
        predictions = model.predict(input_array)
        pred_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
  
        if 0 <= pred_index < len(index_to_label):
            label = index_to_label[pred_index]
        else:
            label = "unknown"

        notes.append("✅ CNN+GRU model prediction")
        return label, confidence, notes
    except Exception as e:
        return "error", 0.0, [str(e)]

