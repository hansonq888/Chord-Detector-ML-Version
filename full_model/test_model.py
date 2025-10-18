import joblib
import librosa
import numpy as np

def extract_enhanced_features(audio_path):
    """Extract comprehensive audio features (same as training)"""
    try:
        # Load audio with consistent parameters
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        
        # Preprocessing
        y = librosa.util.normalize(y)
        
        # Estimate and correct tuning
        tuning_est = librosa.estimate_tuning(y=y, sr=sr)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-tuning_est)
        
        # Remove silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        features = []
        
        # 1. Chroma features (multiple variants)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
        
        # Statistical features from chroma
        features.extend(np.mean(chroma_stft, axis=1))  # Mean chroma STFT
        features.extend(np.std(chroma_stft, axis=1))   # Std chroma STFT
        features.extend(np.max(chroma_stft, axis=1))   # Max chroma STFT
        features.extend(np.mean(chroma_cqt, axis=1))   # Mean chroma CQT
        features.extend(np.std(chroma_cqt, axis=1))    # Std chroma CQT
        features.extend(np.mean(chroma_cens, axis=1))  # Mean chroma CENS
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.max(spectral_centroids)
        ])
        
        # 3. Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])
        
        # 5. MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))  # Mean MFCCs
        features.extend(np.std(mfccs, axis=1))   # Std MFCCs
        
        # 6. Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend(np.mean(tonnetz, axis=1))
        
        # 7. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spectral_contrast, axis=1))
        
        # 8. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # 9. Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.extend([
            np.mean(y_harmonic),
            np.std(y_harmonic),
            np.mean(y_percussive),
            np.std(y_percussive)
        ])
        
        # Ensure consistent feature size
        target_features = 100
        if len(features) < target_features:
            features.extend([0] * (target_features - len(features)))
        elif len(features) > target_features:
            features = features[:target_features]
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros(100, dtype=np.float32)

# Load model data (includes model, scaler, and chord names)
model_data = joblib.load("full_model/model.pkl")
model = model_data['model']
scaler = model_data['scaler']

note_names = [
    "A_major", "A_minor",
    "A#_major", "A#_minor",
    "B_major", "B_minor",
    "C_major", "C_minor",
    "C#_major", "C#_minor",
    "D_major", "D_minor",
    "D#_major", "D#_minor",
    "E_major", "E_minor",
    "F_major", "F_minor",
    "F#_major", "F#_minor",
    "G_major", "G_minor",
    "G#_major", "G#_minor"
]

correct = 0


for chord in note_names:
    audio_path = "Chords/" + chord + ".wav"  # Replace with your audio file's name or path
    features = extract_enhanced_features(audio_path)  # Extract features
    features_scaled = scaler.transform(features.reshape(1, -1))  # Scale features
    predicted_label = model.predict(features_scaled)  # Get prediction from the model
    print("Predicted chord:", predicted_label[0])  # Print result
    print("Actual chord: " + audio_path +"\n")
    if (chord == predicted_label[0]):
        correct += 1
print(str(correct) + "/24 chords correct")

# audio_path = "Sample Tests/C_major_guitar.wav"  # Replace with your audio file's name or path
# features = extract_enhanced_features(audio_path)  # Extract features
# features_scaled = scaler.transform(features.reshape(1, -1))  # Scale features
# predicted_label = model.predict(features_scaled)  # Get prediction from the model
# print("Predicted chord:", predicted_label[0])  # Print result
# print("Actual chord: C_major \n")
# if ("C_major" == predicted_label[0]):
#     correct += 1

# print(str(correct) + "/1 chords correct")
    
