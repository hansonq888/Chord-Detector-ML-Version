import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
AUDIO_DIR = "wav_dataset"
CHORD_NAMES = [
    "A_major", "A_minor", "A#_major", "A#_minor",
    "B_major", "B_minor", "C_major", "C_minor",
    "C#_major", "C#_minor", "D_major", "D_minor",
    "D#_major", "D#_minor", "E_major", "E_minor",
    "F_major", "F_minor", "F#_major", "F#_minor",
    "G_major", "G_minor", "G#_major", "G#_minor"
]

def extract_enhanced_features(audio_path):
    """Extract comprehensive audio features for better chord detection"""
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

print("Loading training data...")
X = []
y = []

# Feature extraction with progress tracking
files_processed = 0
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        filename_base = filename.replace(".wav", "")
        label = "_".join(filename_base.split("_")[:2])
        
        if label in CHORD_NAMES:
            path = os.path.join(AUDIO_DIR, filename)
            features = extract_enhanced_features(path)
            X.append(features)
            y.append(label)
            files_processed += 1
            
            if files_processed % 100 == 0:
                print(f"Processed {files_processed} files...")

print(f"Loaded {files_processed} training samples")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train improved model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train final model
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.3f} ({acc*100:.1f}%)")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=CHORD_NAMES))

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'chord_names': CHORD_NAMES
}
joblib.dump(model_data, "model.pkl")
print("Model saved as model.pkl")
