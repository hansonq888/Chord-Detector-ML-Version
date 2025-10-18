import librosa
import numpy as np

# Load the audio file
y, sr = librosa.load("Sample Tests/C_major_guitar.wav")

# Estimate tuning and correct
tuning_est = librosa.estimate_tuning(y=y, sr=sr)
print("Estimated tuning (in semitones):", tuning_est)
y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-tuning_est)

# Get the chromagram
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
avg_chroma = np.mean(chroma, axis=1)

# Chord templates: 12 pitch classes (C, C#, ..., B)
CHORDS = {
    #      [C  C# D  D# E  F  F# G  G# A  A# B]
    'C':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C E G
    'Cm':  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C D# G

    'C#':  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C# F G#
    'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # C# E G#

    'D':   [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D F# A
    'Dm':  [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # D F A

    'D#':  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D# G A#
    'D#m': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # D# F# A#

    'E':   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # E G# B
    'Em':  [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # E G B

    'F':   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F A C
    'Fm':  [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # F G# C

    'F#':  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # F# A# C#
    'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # F# A C#

    #      [C  C# D  D# E  F  F# G  G# A  A# B]
    'G':   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G B D
    'Gm':  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # G A# D

    'G#':  [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # G# C D#
    'G#m': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  # G# B D#

    'A':   [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A C# E
    'Am':  [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A C E

    'A#':  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A# D F
    'A#m': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A# C# F

    'B':   [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # B D# F#
    'Bm':  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # B D F#
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find best matching chord
best_chord = None
best_score = -1

for name, template in CHORDS.items():
    score = cosine_similarity(avg_chroma, template)
    if score > best_score:
        best_score = score
        best_chord = name

print(f"🎵 Detected Chord: {best_chord} (Score: {best_score:.2f})")
