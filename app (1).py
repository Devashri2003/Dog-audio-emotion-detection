import streamlit as st
import librosa
import numpy as np
import joblib

# Load model
model = joblib.load("dog_emotion_model.pkl")

emotion_map = {0: "Happy", 1: "Hungry", 2: "Sad"}

def extract_features(file):
    audio, sr = librosa.load(file)

    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    dominant_freq = abs(freqs[np.argmax(np.abs(fft))])

    energy = np.mean(audio**2)

    return [dominant_freq, energy]

def predict_emotion(file):
    features = extract_features(file)
    features = np.array(features).reshape(1, -1)

    cluster = model.predict(features)[0]
    emotion = emotion_map.get(cluster, "Unknown")

    return features[0][0], features[0][1], emotion

# UI
st.title("🐶 Dog Emotion Detection")

uploaded_file = st.file_uploader("Upload Dog Audio", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Predict Emotion"):
        freq, energy, emotion = predict_emotion(uploaded_file)

        st.write(f"🎵 Frequency: {freq:.2f} Hz")
        st.write(f"⚡ Energy: {energy:.5f}")
        st.success(f"🐶 Emotion: {emotion}")
