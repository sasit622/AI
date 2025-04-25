import streamlit as st
import librosa
import numpy as np
import joblib
from io import BytesIO

# Load model and preprocessors
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define genre colors for visual feedback
GENRE_COLORS = {
    "blues": "#1f77b4",
    "classical": "#ff7f0e",
    "country": "#2ca02c",
    "disco": "#d62728",
    "hiphop": "#9467bd",
    "jazz": "#8c564b",
    "metal": "#e377c2",
    "pop": "#7f7f7f",
    "reggae": "#bcbd22",
    "rock": "#17becf"
}

def extract_features(file):
    y, sr = librosa.load(file, duration=30)  # Load the first 30 seconds
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        font-family: 'Arial', sans-serif;
        color: #1f77b4;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #4a4a4a;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #135a8d;
    }
    .prediction-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    .expander-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 40px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<h1 class="title">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a music clip (MP3 or WAV) and discover its genre!</p>', unsafe_allow_html=True)

# Layout with columns for better organization
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader widget with enhanced styling
    audio_file = st.file_uploader(
        "🎶 Upload your audio file",
        type=["mp3", "wav"],
        help="Upload a clear MP3 or WAV file (up to 30 seconds recommended)"
    )

    # If audio file is uploaded
    if audio_file:
        # Display audio player
        st.audio(audio_file, format=audio_file.type)

        # Button to trigger prediction
        if st.button("🎧 Predict Genre"):
            with st.spinner("Analyzing your music..."):
                try:
                    # Convert MP3 to WAV if necessary
                    if audio_file.type == 'audio/mp3':
                        audio_bytes = BytesIO(audio_file.read())
                        y, sr = librosa.load(audio_bytes, duration=30)
                    else:
                        y, sr = librosa.load(audio_file, duration=30)

                    # Extract features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    features = np.mean(mfccs.T, axis=0)

                    # Scale the features and make a prediction
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)

                    # Get the predicted genre
                    genre = label_encoder.inverse_transform(prediction)[0]

                    # Display prediction result in a styled box
                    st.markdown(
    f"""
    <div class="prediction-box" style="
        background: linear-gradient(135deg, {GENRE_COLORS.get(genre.lower(), '#f0f0f0')}20, #ffffff);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        ">
        <h3 style="
            color: {GENRE_COLORS.get(genre.lower(), '#333')};
            font-family: 'Arial', sans-serif;
            font-size: 24px;
            margin: 0;
            ">
            Predicted Genre: <b>{genre.capitalize()}</b> 
        </h3>
    </div>
    <style>
        .prediction-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        }}
    </style>
    """,
    unsafe_allow_html=True
)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")



# Expander for additional info with improved styling
with st.expander("ℹ️ About This App", expanded=False):
    st.markdown(
        """
        <style>
            .expander-content {
                font-family: 'Arial', sans-serif;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            .expander-content h4 {
                color: #1f77b4;
                margin-bottom: 10px;
            }
            .expander-content p {
                font-size: 14px;
                color: #333;
                line-height: 1.5;
                margin: 5px 0;
            }
            .expander-content ul {
                margin: 10px 0;
                padding-left: 20px;
            }
            .expander-content li {
                margin-bottom: 5px;
                font-size: 14px;
                color: #333;
            }
            .expander-content b {
                color: #1f77b4;
            }
        </style>
        <div class="expander-content">
            <h4>About the Music Genre Classifier</h4>
            <p>
                This app uses a machine learning model to classify music genres based on audio features, including:
            </p>
            <ul>
                <li><b>Mel-frequency cepstral coefficients (MFCCs)</b>: Captures the timbral aspects of sound.</li>
                <li><b>Chroma features</b>: Represents the harmonic content.</li>
                <li><b>Spectral contrast</b>: Highlights differences in frequency bands.</li>
            </ul>
            <p>
                The model was trained on a diverse dataset of music genres, ensuring robust predictions.
                For optimal results, use high-quality audio samples of 30 seconds or less.
            </p>
            <p style="color: #d62728;">
                <b>Note</b>: Ensure FFmpeg is installed for .mp3 support. Only valid .mp3 or .wav files are accepted.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    '<p class="footer">Built By Sasi Kavin Kabilan    | © 2025 Music Genre Classifier</p>',
    unsafe_allow_html=True
)