import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths for your dataset folders
PETROL_FOLDER = "dataset/petrol"
DIESEL_FOLDER = "dataset/diesel"
SAVE_MODEL_PATH = "fuel_classifier.pkl"

# Use local YAMNet model path, not URL
yamnet_path = "yamnet"
print("[INFO] Loading local YAMNet model...")
yamnet = hub.load(yamnet_path)

def get_embedding(audio_path):
    '''Extract the mean YAMNet embedding for an audio file.'''
    y, sr = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform)
    return np.mean(embeddings.numpy(), axis=0)

def build_dataset():
    features = []
    labels = []
    for label, folder in [('petrol', PETROL_FOLDER), ('diesel', DIESEL_FOLDER)]:
        print(f"[INFO] Processing folder: {folder}")
        for fname in os.listdir(folder):
            if not (fname.endswith('.wav') or fname.endswith('.mp3')):
                continue
            fpath = os.path.join(folder, fname)
            try:
                emb = get_embedding(fpath)
                features.append(emb)
                labels.append(label)
            except Exception as e:
                print(f"[WARN] Could not process {fpath}: {e}")
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    X, y = build_dataset()
    print("[INFO] Dataset size:", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("[RESULT] Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, SAVE_MODEL_PATH)
    print(f"[INFO] Saved classifier at {SAVE_MODEL_PATH}")
