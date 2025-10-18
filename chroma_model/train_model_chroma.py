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

def extract_chroma_features(audio_path):
    """Extract only chroma features for chord detection"""
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
        
        # Skip if audio is too short after trimming
        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            return np.zeros(12, dtype=np.float32)
        
        # Extract chroma features (multiple variants for robustness)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
        
        # Combine different chroma variants for better robustness
        # Use mean of all three chroma types
        chroma_combined = (np.mean(chroma_stft, axis=1) + 
                          np.mean(chroma_cqt, axis=1) + 
                          np.mean(chroma_cens, axis=1)) / 3
        
        # Normalize to make features more consistent
        chroma_combined = chroma_combined / (np.sum(chroma_combined) + 1e-8)
        
        return chroma_combined.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros(12, dtype=np.float32)

print("Loading training data with chroma features only...")
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
            features = extract_chroma_features(path)
            X.append(features)
            y.append(label)
            files_processed += 1
            
            if files_processed % 1000 == 0:
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

# Train model optimized for chroma features
print("Training Random Forest model on chroma features...")
model = RandomForestClassifier(
    n_estimators=200,  # Fewer trees since we have fewer features
    max_depth=15,      # Reduced depth
    min_samples_split=2,
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

# Feature importance analysis
feature_names = [f"Chroma_{i}" for i in range(12)]
feature_importance = model.feature_importances_
print(f"\nFeature Importance (Chroma features):")
for i, importance in enumerate(feature_importance):
    note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][i]
    print(f"  {note_name}: {importance:.3f}")

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'chord_names': CHORD_NAMES,
    'feature_type': 'chroma_only'
}
joblib.dump(model_data, "model_chroma.pkl")
print("Chroma-only model saved as model_chroma.pkl")

# Compare with template-based approach
print("\n" + "="*50)
print("COMPARISON WITH TEMPLATE-BASED APPROACH")
print("="*50)

# Load a sample file to test template matching
sample_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')][:10]
template_accuracy = 0

for sample_file in sample_files:
    try:
        path = os.path.join(AUDIO_DIR, sample_file)
        y, sr = librosa.load(path, sr=22050, duration=3.0)
        y = librosa.util.normalize(y)
        tuning_est = librosa.estimate_tuning(y=y, sr=sr)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-tuning_est)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < sr * 0.5:
            continue
            
        # Template matching approach
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        avg_chroma = np.mean(chroma, axis=1)
        
        # Chord templates
        CHORDS = {
            'C':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'Cm':  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'C#':  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'D':   [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'Dm':  [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'D#':  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'D#m': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            'E':   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            'Em':  [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'F':   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Fm':  [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'F#':  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'G':   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Gm':  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            'G#':  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            'G#m': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            'A':   [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'Am':  [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'A#':  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'A#m': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'B':   [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            'Bm':  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        }
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        best_chord = None
        best_score = -1
        
        for name, template in CHORDS.items():
            score = cosine_similarity(avg_chroma, template)
            if score > best_score:
                best_score = score
                best_chord = name
        
        # Get true label
        filename_base = sample_file.replace(".wav", "")
        true_label = "_".join(filename_base.split("_")[:2])
        
        if best_chord == true_label:
            template_accuracy += 1
            
    except Exception as e:
        continue

template_accuracy = template_accuracy / len(sample_files)
print(f"Template matching accuracy (sample): {template_accuracy:.3f} ({template_accuracy*100:.1f}%)")
print(f"ML model accuracy: {acc:.3f} ({acc*100:.1f}%)")

if acc > template_accuracy:
    print("ML model performs better than template matching!")
else:
    print("Template matching performs better - consider using template approach")
