import streamlit as st
import librosa
import numpy as np
import torch
from torch import nn
from pydub import AudioSegment
import uuid
import os
import subprocess

# ------------------------------------------------------------------------------
# Define your CNN architecture first (same as during training)

class CNN(nn.Module):
    def __init__(self, n_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 62, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.conv1(x)  # [B, 16, 20, 250]
        x = self.conv2(x)  # [B, 32, 10, 125]
        x = self.conv3(x)  # [B, 64, 5, 62]
        x = self.flatten(x)  # [B, 64*5*62]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ------------------------------------------------------------------------------
# Audio Conversion

def extract_audio(video_file, audio_file):
    """Extract audio from video using ffmpeg."""
    command = ['ffmpeg', '-i', video_file, '-q:a', '0', '-ac', '1', '-ar', '16000', '-y', audio_file]
    subprocess.call(command)

def convert_to_wav(file):
    """Save uploaded file and convert it to .wav format if it's not already."""
    input_file = f"temp_{uuid.uuid4()}.input"
    output_file = f"temp_{uuid.uuid4()}.wav"

    with open(input_file, "wb") as f:
        f.write(file.read())  # Streamlit's UploadedFile is a file-like object
    
    # Get file extension
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext == '.mp4':
        extract_audio(input_file, output_file)
    else:
        try:
            # Handle all audio formats including opus
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_file, format='wav')
        except:
            # Fallback to ffmpeg for formats that pydub might not handle directly
            command = ['ffmpeg', '-i', input_file, '-ac', '1', '-ar', '16000', '-y', output_file]
            subprocess.call(command)

    os.remove(input_file)  # Clean up
    return output_file


# ---------------------------------------------------------------------------
# Audio Preprocessing (Mel)

def extract_features(filename, n_mels=40, sr=16000, max_length=500):
    """
    Computes melspectrogram and converts it into a padded array.
    """
    audio, sr = librosa.load(filename, sr=sr)
    audio = librosa.util.normalize(audio)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < max_length:
        pad = max_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :max_length]

    return mel_db

# ---------------------------------------------------------------------------
# Streamlit UI

st.title("Deepfake Voice Detection")
st.write("Classify whether an audio is real or fake.")
uploaded_file = st.file_uploader("Choose an audio or video file", 
                               type=['wav', 'mp3', 'ogg', 'm4a', 'mp4', 'opus'])

if uploaded_file is not None:
    wav_file = convert_to_wav(uploaded_file)

    mel = extract_features(wav_file)
    mel = np.expand_dims(mel, 0)  # adding channel
    mel = np.expand_dims(mel, 0)  # adding batch
    mel = mel.astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    mel_tensor = torch.tensor(mel, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(mel_tensor)
        predicted = preds.argmax(1).item()

    label = "Fake" if predicted == 1 else "Real"

    st.success(f"The audio is classified as: {label}")

    os.remove(wav_file)  # Clean up