import pyaudio
import numpy as np
import librosa
import joblib
import threading
import time
import queue
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class LiveChordDetector:
    def __init__(self, model_path="model.pkl", chunk_duration=1.0, overlap=0.5):
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sample_rate = 22050
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap)
        
        # Load the trained model
        print("Loading trained model...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.chord_names = model_data['chord_names']
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Audio buffer for overlapping chunks
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 3))  # 3 second buffer
        self.audio_queue = queue.Queue()
        
        # Detection results
        self.current_chord = "No chord detected"
        self.confidence = 0.0
        self.detection_history = deque(maxlen=10)  # Keep last 10 detections
        
        print("Live chord detector initialized!")
        print(f"Chunk duration: {chunk_duration}s, Overlap: {overlap}s")
        print(f"Sample rate: {self.sample_rate} Hz")
        print("Available chords:", ", ".join(self.chord_names))
    
    def extract_enhanced_features(self, audio_data):
        """Extract features from audio data (same as training)"""
        try:
            # Convert to float32 and normalize
            y = audio_data.astype(np.float32)
            y = librosa.util.normalize(y) # normalizes amplitude to 1
            
            # Estimate and correct tuning
            tuning_est = librosa.estimate_tuning(y=y, sr=self.sample_rate) # outputs how off tuning is in cents
            y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=-tuning_est) # shifts recording tuning
            
            # Remove silence
            y, _ = librosa.effects.trim(y, top_db=20) # filter out lower db sound
            
            # Skip if audio is too short after trimming
            if len(y) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                return None
            
            features = []
            
            # 1. Chroma features (multiple variants)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=self.sample_rate, n_chroma=12)
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=self.sample_rate, n_chroma=12)
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=self.sample_rate, n_chroma=12)
            
            # Statistical features from chroma
            features.extend(np.mean(chroma_stft, axis=1))  # Mean chroma STFT
            features.extend(np.std(chroma_stft, axis=1))   # Std chroma STFT
            features.extend(np.max(chroma_stft, axis=1))   # Max chroma STFT
            features.extend(np.mean(chroma_cqt, axis=1))   # Mean chroma CQT
            features.extend(np.std(chroma_cqt, axis=1))    # Std chroma CQT
            features.extend(np.mean(chroma_cens, axis=1))  # Mean chroma CENS
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.max(spectral_centroids)
            ])
            
            # 3. Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)[0]
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
            mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))  # Mean MFCCs
            features.extend(np.std(mfccs, axis=1))   # Std MFCCs
            
            # 6. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=self.sample_rate)
            features.extend(np.mean(tonnetz, axis=1))
            
            # 7. Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)
            features.extend(np.mean(spectral_contrast, axis=1))
            
            # 8. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sample_rate)
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
            print(f"Error extracting features: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Add to queue for processing
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio and detect chord"""
        features = self.extract_enhanced_features(audio_chunk)
        
        if features is not None:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction and confidence
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Update current detection
            self.current_chord = prediction
            self.confidence = confidence
            
            # Add to history
            self.detection_history.append({
                'chord': prediction,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            return prediction, confidence
        
        return None, 0.0
    
    def start_recording(self):
        """Start live audio recording and chord detection"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            print("🎵 Live chord detection started!")
            print("Play some chords and watch the detection...")
            print("Press Ctrl+C to stop")
            print("-" * 50)
            
            # Create the processing head (runs at the same time as main, like forking in background)
            # Thread runs _processing_loop
            # Daemon Thread ends when main function ends
            processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            processing_thread.start() # Starts the thread
            
            # Main display loop
            self._display_loop()
            
        except Exception as e:
            print(f"Error starting recording: {e}")
        finally:
            self.stop_recording()
    
    def _processing_loop(self):
        # Background thread that processes audio chunks
        while self.is_recording:
            try:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    # Create overlapping chunk from buffer
                    if len(self.audio_buffer) >= self.chunk_size:
                        # Get the most recent chunk with overlap
                        start_idx = max(0, len(self.audio_buffer) - self.chunk_size)
                        chunk = np.array(list(self.audio_buffer)[start_idx:])
                        
                        # Process the chunk
                        self.process_audio_chunk(chunk)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def _display_loop(self):
        """Main display loop for showing detected chords"""
        last_chord = ""
        last_confidence = 0.0
        
        try:
            while self.is_recording:
                # Only update display if chord changed or confidence changed significantly
                if (self.current_chord != last_chord or 
                    abs(self.confidence - last_confidence) > 0.05):
                    
                    # Clear line and show current detection
                    print(f"\r🎵 Current Chord: {self.current_chord:<12} "
                          f"Confidence: {self.confidence:.2f} "
                          f"Buffer: {len(self.audio_buffer)/self.sample_rate:.1f}s", end="", flush=True)
                    
                    last_chord = self.current_chord
                    last_confidence = self.confidence
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nStopping chord detection...")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop live audio recording"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("\n🎵 Live chord detection stopped!")
        self._show_detection_summary()
    
    def _show_detection_summary(self):
        """Show summary of detected chords"""
        if not self.detection_history:
            print("No chords detected.")
            return
        
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        
        # Count chord occurrences
        chord_counts = {}
        for detection in self.detection_history:
            chord = detection['chord']
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
        
        # Sort by frequency
        sorted_chords = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("Most frequently detected chords:")
        for chord, count in sorted_chords[:5]:
            percentage = (count / len(self.detection_history)) * 100
            print(f"  {chord}: {count} times ({percentage:.1f}%)")
        
        print(f"\nTotal detections: {len(self.detection_history)}")
        print("="*50)

def main():
    """Main function to run the live chord detector"""
    print("🎵 Live Chord Detector")
    print("=" * 30)
    
    # Check if model exists
    try:
        detector = LiveChordDetector()
    except FileNotFoundError:
        print("Error: model.pkl not found!")
        print("Please run train_model.py first to train the model.")
        return
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    try:
        detector.start_recording()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    main()
