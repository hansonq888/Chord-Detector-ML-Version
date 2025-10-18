# Chord Detector - File Structure

## Project Organization

The project is now organized into clear folders for different model types:

### Root Directory
- `live_chord_detector_gui.py` - GUI for Full Model (100 features)
- `live_chord_detector_chroma_gui.py` - GUI for Chroma Model (12 features)
- `Chords/` - Test chord audio files
- `wav_dataset/` - Training dataset
- `generate_audio/` - Audio generation scripts
- `midi_chords/` - MIDI chord files
- `Sample Tests/` - Sample test files

### Full Model Folder (`full_model/`)
Contains the comprehensive 100-feature model:
- `live_chord_detector.py` - Live detector for full model
- `model.pkl` - Trained full model (100 features)
- `train_model.py` - Training script for full model
- `test_model.py` - Testing script for full model

### Chroma Model Folder (`chroma_model/`)
Contains the chroma-only model:
- `live_chord_detector_chroma.py` - Live detector for chroma model
- `model_chroma.pkl` - Trained chroma model (12 features)
- `train_model_chroma.py` - Training script for chroma model
- `test_model_chroma.py` - Testing script for chroma model

## Usage

### Full Model (100 Features)
```bash
# Run GUI
python live_chord_detector_gui.py

# Run command line detector
python full_model/live_chord_detector.py

# Test model
python full_model/test_model.py

# Train model
python full_model/train_model.py
```

### Chroma Model (12 Features)
```bash
# Run GUI
python live_chord_detector_chroma_gui.py

# Run command line detector
python chroma_model/live_chord_detector_chroma.py

# Test model
python chroma_model/test_model_chroma.py

# Train model
python chroma_model/train_model_chroma.py
```

## Model Differences

- **Full Model**: Uses 100 features including chroma, spectral, MFCC, tempo, and harmonic features
- **Chroma Model**: Uses only 12 chroma features for more focused chord detection

Both models achieve 100% accuracy on the test chord files.
