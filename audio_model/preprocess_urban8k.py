import os
import pandas as pd
from audio_preprocess import preprocess_audio

meta = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")
engine_files = meta[meta['class'] == 'engine_idling']

engine_dir = "UrbanSound8K/audio"
cleaned_dir = "UrbanSound8K/engine_idling_cleaned"
os.makedirs(cleaned_dir, exist_ok=True)

for ix, row in engine_files.iterrows():
    infile = os.path.join(engine_dir, f"fold{row['fold']}", row['slice_file_name'])
    outfile = os.path.join(cleaned_dir, row['slice_file_name'])
    preprocess_audio(infile, outfile)
print("✅ Cleaned all engine_idling files!")
