import os
import json
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, GRU, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from audio_preprocess import preprocess_audio

TRAIN_DIR = "ai-mechanic-export/training"
TEMP_DIR = "temp_audio_cleaned"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_label(filename):
    return filename.split(".")[0].strip().lower()

def pad_mfcc(mfcc, max_frames=100):
    if mfcc.shape[0] > max_frames:
        return mfcc[:max_frames, :]
    else:
        pad_amt = max_frames - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad_amt), (0,0)), mode='constant')

X, y = [], []
label_set = set()

print(f"Reading labels from directory: {TRAIN_DIR}")

for fname in os.listdir(TRAIN_DIR):
    if fname.endswith(".wav"):
        label_set.add(extract_label(fname))
label_list = sorted(label_set)
label_to_index = {label: idx for idx, label in enumerate(label_list)}
print(f"Found labels: {label_list}")

for fname in os.listdir(TRAIN_DIR):
    if fname.endswith(".wav"):
        try:
            label = extract_label(fname)
            audio_path = os.path.join(TRAIN_DIR, fname)
            temp_path = os.path.join(TEMP_DIR, fname)
            noise_temp_path = os.path.join(TEMP_DIR, f"noise_{fname}")
            preprocess_audio(audio_path, temp_path, noise_temp_path)
            y_audio, sr = librosa.load(temp_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
            mfcc = mfcc.T 
            mfcc_padded = pad_mfcc(mfcc, 100)
            X.append(mfcc_padded)
            y.append(label_to_index[label])
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

X = np.array(X)
y_cat = to_categorical(np.array(y), num_classes=len(label_list))
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# --- Define CNN+GRU model ---
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(100, 40)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),
    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.2),
    GRU(48, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_list), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=35, batch_size=32, validation_data=(X_val, y_val))

# --- Save model/labels ---
model.save("audio_cnn_gru_model.h5")
with open("cnn_gru_labels.json", "w") as f:
    json.dump(label_to_index, f)

best_val_acc = max(history.history['val_accuracy'])
with open("cnn_gru_last_accuracy.txt", "w") as f:
    f.write(str(best_val_acc))

print(f"Best validation accuracy: {best_val_acc:.4f} saved to audio_model/cnn_gru_last_accuracy.txt")

# --- Accuracy curve ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN+GRU Training & Validation Accuracy by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()