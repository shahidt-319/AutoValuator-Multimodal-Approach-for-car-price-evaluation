import librosa
import scipy.signal
import noisereduce as nr
import soundfile as sf

# bandpass filter
def bandpass_filter(y, sr, low=40, high=5000):
    
    sos = scipy.signal.butter(10, [low, high], btype='band', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, y)


#Noise reduction
def reduce_noise(y, sr, noise_clip=None, noise_clip_sec=0.5):
   
    if noise_clip is None:
        n_samples = int(noise_clip_sec * sr)
        noise_clip = y[:n_samples]
    reduced = nr.reduce_noise(y=y, y_noise=noise_clip, sr=sr)
    return reduced, noise_clip


# Audio preprocessing
def preprocess_audio(input_path, output_clean_path, output_noise_path, sr_target=22050, noise_clip=None):
   
    # Load audio
    y, sr = librosa.load(input_path, sr=sr_target)
    y_filtered = bandpass_filter(y, sr)
    y_denoised, noise_clip_out = reduce_noise(y_filtered, sr, noise_clip=noise_clip)
    sf.write(output_clean_path, y_denoised, sr)
    sf.write(output_noise_path, noise_clip_out, sr)
    return output_clean_path, output_noise_path
