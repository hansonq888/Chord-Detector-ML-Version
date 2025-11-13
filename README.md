# Chord Detector - ML

A machine learning-based chord detection system that can identify 24 different chords (12 major and 12 minor) in real-time from audio input. The project includes two models: a full-featured model using 100 audio features and a lightweight chroma-only model using 12 chroma features.

## Features

- **Real-time chord detection** from microphone input
- **24 chord types** supported (A, A#, B, C, C#, D, D#, E, F, F#, G, G# - each in major and minor)
- **Two model variants**:
  - **Full Model**: Uses 100 comprehensive audio features (chroma, spectral, MFCC, tonnetz, etc.)
  - **Chroma Model**: Uses only 12 chroma features for faster, lighter-weight detection
- **Multiple interfaces**:
  - Tkinter GUI (desktop application)
  - Gradio web interface
  - Streamlit web interface
  - Command-line interface
- **Dataset generation tools** for creating training data from MIDI files
- **Executable deployment** for easy distribution

## Project Structure

```
Chord Detector - ML/
├── chroma_model/              # Chroma-only model files
│   ├── train_model_chroma.py  # Training script for chroma model
│   ├── test_model_chroma.py   # Testing script for chroma model
│   ├── live_chord_detector_chroma.py  # Live detection class
│   └── model_chroma.pkl       # Trained chroma model
├── full_model/                # Full-featured model files
│   ├── train_model.py         # Training script for full model
│   ├── test_model.py          # Testing script for full model
│   ├── live_chord_detector.py # Live detection class
│   └── model.pkl              # Trained full model
├── generate_audio/            # Dataset generation scripts
│   ├── generate_audio_variations_inversions.py  # Main generation script
│   ├── generate_audio_variations.py
│   └── generate_midi.py
├── Chords/                    # Reference chord audio files
├── midi_chords/               # Source MIDI files for dataset generation
├── wav_dataset/               # Generated training dataset (22,680 files)
├── Sample Tests/              # Test audio files
├── live_chord_detector_gui.py # Full model GUI
├── live_chord_detector_chroma_gui.py  # Chroma model GUI
├── live_chord_detector_gradio.py      # Gradio interface
├── live_chord_detector_streamlit.py   # Streamlit interface
├── chord_detector.py          # Simple template-based detector
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows (for executable deployment)
- FluidSynth (for dataset generation)
- A soundfont file (e.g., GeneralUser-GS.sf2)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Chord Detector - ML"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or use the installation script:
   ```bash
   python install_requirements.py
   ```

4. **Install FluidSynth** (for dataset generation):
   - Download from [FluidSynth website](https://www.fluidsynth.org/)
   - Ensure `fluidsynth` is in your system PATH
   - Place a soundfont file (`.sf2`) in a known location and update the path in generation scripts

## Usage

### Training Models

#### Train Full Model (100 features)
```bash
cd full_model
python train_model.py
```

This will:
- Load audio files from `wav_dataset/`
- Extract 100 features per file (chroma, spectral, MFCC, tonnetz, etc.)
- Train a Random Forest classifier
- Save the model as `model.pkl`

#### Train Chroma Model (12 features)
```bash
cd chroma_model
python train_model_chroma.py
```

This will:
- Load audio files from `wav_dataset/`
- Extract 12 chroma features per file
- Train a Random Forest classifier optimized for chroma features
- Save the model as `model_chroma.pkl`

### Testing Models

#### Test Full Model
```bash
cd full_model
python test_model.py
```

#### Test Chroma Model
```bash
cd chroma_model
python test_model_chroma.py
```

### Running the GUI Applications

#### Full Model GUI
```bash
python live_chord_detector_gui.py
```

#### Chroma Model GUI
```bash
python live_chord_detector_chroma_gui.py
```

Both GUIs provide:
- Real-time chord detection from microphone
- Confidence score display
- Detection history
- Start/Stop controls

### Web Interfaces

#### Gradio Interface
```bash
python live_chord_detector_gradio.py
```

#### Streamlit Interface
```bash
streamlit run live_chord_detector_streamlit.py
```

### Dataset Generation

To generate the training dataset from MIDI files:

1. **Prepare MIDI files**: Place your MIDI chord files in the `midi_chords/` folder

2. **Configure settings**: Edit `generate_audio/generate_audio_variations_inversions.py`:
   - Set `SOUNDFONT` path to your soundfont file
   - Adjust parameters (pitch shifts, velocity multipliers, octave shifts, inversions)

3. **Run generation**:
   ```bash
   python generate_audio/generate_audio_variations_inversions.py
   ```

   This will generate approximately 22,680 WAV files with various transformations:
   - 24 base chords
   - 3 octave shifts (-12, 0, +12 semitones)
   - 3 inversions (0, 1, 2)
   - 21 pitch shifts (-0.10 to +0.10 semitones)
   - 5 velocity multipliers (0.85, 0.95, 1.0, 1.05, 1.15)

## Model Details

### Full Model
- **Features**: 100 audio features including:
  - Chroma features (STFT, CQT, CENS variants with statistics)
  - Spectral features (centroid, rolloff, contrast)
  - MFCC coefficients (13 coefficients with statistics)
  - Tonnetz features
  - Zero crossing rate
  - Harmonic and percussive components
  - Tempo estimation
- **Algorithm**: Random Forest Classifier
- **Parameters**: 300 estimators, max depth 20
- **Use case**: Maximum accuracy, suitable for offline analysis

### Chroma Model
- **Features**: 12 chroma features (one per pitch class)
  - Combines chroma_stft, chroma_cqt, and chroma_cens
  - Normalized and averaged
- **Algorithm**: Random Forest Classifier
- **Parameters**: 200 estimators, max depth 15
- **Use case**: Real-time detection, lightweight, fast inference

## Deployment

### Building Executables

#### Chroma Model Executable
```bash
python build_chroma_exe.py
```

Or use the batch file:
```bash
build_chroma.bat
```

This creates:
- `dist/ChordDetectorChroma_Package/ChordDetectorChroma.exe`
- `dist/ChordDetectorChroma_Package.zip` (distribution package)

#### Full Model Executable
Similar process using PyInstaller with the appropriate spec file.

### Distribution Package Contents

The executable package includes:
- The compiled `.exe` file
- Required model files (`model.pkl` or `model_chroma.pkl`)
- Sample test files
- README with usage instructions
- Installation script (if needed)

## Requirements

Key dependencies:
- `librosa` - Audio analysis and feature extraction
- `scikit-learn` - Machine learning models
- `numpy` - Numerical computations
- `PyAudio` - Real-time audio I/O
- `joblib` - Model serialization
- `tkinter` - GUI framework (usually included with Python)
- `gradio` - Web interface (optional)
- `streamlit` - Web interface (optional)
- `mido` - MIDI file processing (for dataset generation)
- `pyinstaller` - Executable creation (for deployment)

See `requirements.txt` for the complete list.

## How It Works

1. **Audio Input**: Captures audio from microphone in real-time (1-second chunks with 0.5-second overlap)

2. **Preprocessing**:
   - Normalize audio levels
   - Estimate and correct tuning
   - Remove silence

3. **Feature Extraction**:
   - **Full Model**: Extracts 100 comprehensive features
   - **Chroma Model**: Extracts 12 chroma features representing pitch classes

4. **Classification**:
   - Features are scaled using the trained scaler
   - Random Forest model predicts the chord
   - Confidence score is calculated

5. **Output**: Displays detected chord and confidence score

## Troubleshooting

### Audio Input Issues
- Ensure microphone permissions are granted
- Check that PyAudio is correctly installed
- Try adjusting the chunk size in the detection class

### Model Not Found
- Ensure model files (`model.pkl` or `model_chroma.pkl`) exist in the correct directories
- Check file paths in the detection scripts

### Dataset Generation Errors
- Verify FluidSynth is installed and in PATH
- Check soundfont file path is correct
- Ensure MIDI files are valid and in the correct format

### Executable Issues
- Ensure all dependencies are included in the PyInstaller spec file
- Check that model files are included as data files
- Verify hidden imports are specified

## Contributing

Contributions are welcome! Areas for improvement:
- Additional chord types (7th chords, suspended chords, etc.)
- Better feature engineering
- Model architecture improvements
- UI/UX enhancements
- Cross-platform support

## License

[Specify your license here]

## Acknowledgments

- Uses `librosa` for audio analysis
- Uses `scikit-learn` for machine learning
- FluidSynth for MIDI to audio conversion

## Contact

[Your contact information]

