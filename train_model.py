import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Path to your dataset folder
dataset_path = r"C:\Users\sasit\OneDrive\Desktop\ai project\archive\Data\genres_original"

def extract_features(file):
    """
    Extract MFCCs (Mel-frequency cepstral coefficients) from an audio file.
    """
    try:
        # Load the first 30 seconds of the audio file
        y, sr = librosa.load(file, duration=30, mono=True)
        # Check if the audio is valid (non-empty)
        if len(y) == 0:
            print(f"Empty audio file: {file}")
            return None
        # Extract 13 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Return the mean of MFCCs across time
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None

# Load data
features = []
labels = []
genres = os.listdir(dataset_path)

# Loop over each genre folder
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    if not os.path.isdir(genre_path):
        continue
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            file_path = os.path.join(genre_path, file)
            mfccs_mean = extract_features(file_path)
            if mfccs_mean is not None:
                features.append(mfccs_mean)
                labels.append(genre)

# Check if any features were extracted
if not features:
    raise ValueError("No valid audio files were processed. Please check the dataset.")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode the labels (convert genres to integers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model, scaler, and label encoder as .pkl files
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')