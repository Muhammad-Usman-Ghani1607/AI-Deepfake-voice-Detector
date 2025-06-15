🔊 Deepfake Voice Detection
This project is a Streamlit application that uses a Convolutional Neural Network (CNN) to classify whether a given audio or video file contains a real or fake (AI-synthesized) voice.

📌 Features
✅ File Conversion:
Converts uploaded files (audio or video) into a standardized mono, 16 kHz .wav format using pydub and ffmpeg.

✅ Mel-Spectrogram Extraction:
Transforms audio signals into Mel-Spectrograms with librosa and converts them into decibel scale.

✅ CNN Model:
A custom PyTorch Convolutional Neural Network (CNN) is trained to distinguish real from fake based on Mel-Spectrogram patterns.

✅ Streamlit UI:
Provides a simple and interactive UI for uploading files and viewing results directly in the browser.

✅ Clean-up:
Removes temporary files after processing to keep storage clean.

🛠 Tech Stack
Python 3.x

Framework: Streamlit

Deep Learning: PyTorch

Signal Processing: Librosa, pydub, ffmpeg

Other: NumpY for numerical operations

🚀 How To Run
streamlit run app.py

➥ Step 1: Upload an audio or video file (wav, mp3, ogg, m4a, opus, or mp4).

➥ Step 2: The application converts it to a standardized .wav.

➥ Step 3: Mel-spectrogram is extracted and fed into the trained CNN.

➥ Step 4: The UI displays whether the voice is Real or Fake.

✅ This pipeline lets you quickly and accurately classify deepfake audio files with a friendly UI.

NOTE : The training code is provided in the ipynb file.
