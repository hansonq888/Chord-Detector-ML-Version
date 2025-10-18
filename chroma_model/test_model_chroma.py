import joblib
import librosa
import numpy as np

def extract_chroma_features(audio_path):
    """Extract only chroma features (same as training)"""
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

# Load model data (includes model, scaler, and chord names)
model_data = joblib.load("chroma_model/model_chroma.pkl")
model = model_data['model']
scaler = model_data['scaler']

# Test with files in Chords folder
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
total = len(note_names)

print("Testing chroma model with Chords folder files:")
print("=" * 50)

for chord in note_names:
    audio_path = f"Chords/{chord}.wav"
    print(f"\nTesting: {chord}.wav")
    print(f"Expected: {chord}")
    
    features = extract_chroma_features(audio_path)  # Extract features
    features_scaled = scaler.transform(features.reshape(1, -1))  # Scale features
    predicted_label = model.predict(features_scaled)  # Get prediction from the model
    print(f"Predicted: {predicted_label[0]}")
    
    if chord == predicted_label[0]:
        correct += 1
        print("CORRECT")
    else:
        print("INCORRECT")

print(f"\n" + "=" * 50)
print(f"Results: {correct}/{total} chords correct ({correct/total*100:.1f}%)")

# audio_path = "Sample Tests/C_major_guitar.wav"  # Replace with your audio file's name or path
# features = extract_chroma_features(audio_path)  # Extract features
# features_scaled = scaler.transform(features.reshape(1, -1))  # Scale features
# predicted_label = model.predict(features_scaled)  # Get prediction from the model
# print("Predicted chord:", predicted_label[0])  # Print result
# print("Actual chord: C_major \n")
# if ("C_major" == predicted_label[0]):
#     correct += 1

# print(str(correct) + "/1 chords correct")
