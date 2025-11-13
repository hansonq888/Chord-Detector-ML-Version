import os
import mido
import subprocess
import random
import time
from mido import MidiFile, MidiTrack, Message

MIDI_DIR = "midi_chords"
WAV_DIR = "wav_dataset"
SOUNDFONT = r"C:\Soundfonts\GeneralUser-GS.sf2"
PITCH_SHIFTS = [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

VELOCITY_MULTIPLIERS = [0.85, 0.95, 1.0, 1.05, 1.15]
OCTAVE_SHIFTS = [-12, 0, 12]
INVERSIONS = [0, 1, 2]

os.makedirs(WAV_DIR, exist_ok=True)

EXPECTED_FILES = 24 * 3 * 3 * 21 * 5
files_generated = 0
files_failed = 0
start_time = time.time()

def get_note_ons(track):
    return [msg for msg in track if msg.type == "note_on"]

def apply_inversion(note_ons, inversion):
    if inversion == 0:
        return note_ons
    sorted_notes = sorted(note_ons, key=lambda x: x.note)
    for i in range(inversion):
        sorted_notes[i].note += 12
    return sorted(sorted_notes, key=lambda x: x.note)

def modify_midi(input_path, octave_shift, velocity_multiplier, inversion):
    try:
        midi = MidiFile(input_path)
        new_midi = MidiFile()
        for track in midi.tracks:
            new_track = MidiTrack()
            note_ons = []
            for msg in track:
                if msg.type == "note_on":
                    note_ons.append(msg)
            if note_ons:
                note_ons = apply_inversion(note_ons, inversion)
            for msg in track:
                if msg.type in ["note_on", "note_off"]:
                    match = next((n for n in note_ons if n.time == msg.time and n.note % 12 == msg.note % 12), None)
                    note = match.note if match else msg.note
                    note = max(0, min(127, note + octave_shift))
                    if msg.type == "note_on":
                        vel_variation = random.uniform(0.85, 1.15)
                        velocity = int(min(127, max(1, msg.velocity * velocity_multiplier * vel_variation)))
                    else:
                        velocity = msg.velocity
                    new_track.append(msg.copy(note=note, velocity=velocity))
                else:
                    new_track.append(msg)
            new_midi.tracks.append(new_track)
        temp_path = os.path.join(WAV_DIR, f"temp_{random.randint(10000, 99999)}.mid")
        new_midi.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"Error modifying MIDI {input_path}: {e}")
        return None

def pitch_shift_wav(input_wav, output_wav, n_steps):
    try:
        import librosa
        import soundfile as sf
        y, sr = librosa.load(input_wav, sr=None)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        sf.write(output_wav, y_shifted, sr)
        return True
    except Exception as e:
        print(f"Error pitch shifting {input_wav}: {e}")
        return False

def render_midi_to_wav(midi_path, wav_path):
    try:
        command = [
            "fluidsynth",
            "-ni",
            "-T", "wav",
            "-F", wav_path,
            SOUNDFONT,
            midi_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
            return True
        else:
            print(f"FluidSynth created empty file: {wav_path}")
            return False
    except subprocess.TimeoutExpired:
        print(f"FluidSynth timeout for {midi_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"FluidSynth error for {midi_path}: {e}")
        return False
    except Exception as e:
        print(f"Error rendering {midi_path}: {e}")
        return False

print(f"Starting generation of {EXPECTED_FILES} files...")
print("Progress will be shown every 1000 files")

for file in os.listdir(MIDI_DIR):
    if file.endswith(".mid"):
        chord_name = file.replace(".mid", "")
        input_midi = os.path.join(MIDI_DIR, file)
        print(f"Processing {chord_name}...")
        for inversion in INVERSIONS:
            for octave_shift in OCTAVE_SHIFTS:
                for pitch_shift in PITCH_SHIFTS:
                    for velocity_multiplier in VELOCITY_MULTIPLIERS:
                        try:
                            pitch_tag = f"{'n' if pitch_shift < 0 else 'p'}{abs(pitch_shift):.1f}".replace(".", "")
                            velocity_tag = f"v{int(velocity_multiplier*100)}"
                            octave_tag = f"o{octave_shift}"
                            inv_tag = f"inv{inversion}"
                            output_name = f"{chord_name}_{inv_tag}_{octave_tag}_ps{pitch_tag}_{velocity_tag}.wav"
                            output_wav = os.path.join(WAV_DIR, output_name)

                            temp_midi = modify_midi(
                                input_midi,
                                octave_shift=octave_shift,
                                velocity_multiplier=velocity_multiplier,
                                inversion=inversion
                            )
                            
                            if temp_midi is None:
                                files_failed += 1
                                continue

                            temp_wav = os.path.join(WAV_DIR, f"temp_{random.randint(10000, 99999)}.wav")
                            if not render_midi_to_wav(temp_midi, temp_wav):
                                if os.path.exists(temp_midi):
                                    os.remove(temp_midi)
                                files_failed += 1
                                continue

                            if not pitch_shift_wav(temp_wav, output_wav, pitch_shift):
                                if os.path.exists(temp_midi):
                                    os.remove(temp_midi)
                                if os.path.exists(temp_wav):
                                    os.remove(temp_wav)
                                files_failed += 1
                                continue

                            if os.path.exists(temp_midi):
                                os.remove(temp_midi)
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)
                            
                            files_generated += 1
                            if files_generated % 1000 == 0:
                                elapsed = time.time() - start_time
                                rate = files_generated / elapsed
                                remaining = (EXPECTED_FILES - files_generated) / rate if rate > 0 else 0
                                print(f"Generated {files_generated}/{EXPECTED_FILES} files ({files_generated/EXPECTED_FILES*100:.1f}%) - Rate: {rate:.1f} files/sec - ETA: {remaining/60:.1f} min")
                                
                        except Exception as e:
                            print(f"Error processing {chord_name} variation: {e}")
                            files_failed += 1
                            continue

elapsed = time.time() - start_time
print(f"\n" + "="*60)
print("GENERATION COMPLETE")
print("="*60)

actual_files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]
final_count = len(actual_files)

print(f"Files generated: {files_generated}")
print(f"Files failed: {files_failed}")
print(f"Expected files: {EXPECTED_FILES}")
print(f"Actual files in directory: {final_count}")
print(f"Success rate: {files_generated/EXPECTED_FILES*100:.1f}%")
print(f"Time elapsed: {elapsed/60:.1f} minutes")
print(f"Average rate: {files_generated/elapsed:.1f} files/second")

if final_count == EXPECTED_FILES:
    print(f"\nSUCCESS! All {EXPECTED_FILES} files generated!")
else:
    print(f"\nGeneration incomplete: {final_count}/{EXPECTED_FILES} files")

print("All chord variations (inversions, pitch, velocity, octave) generated.")
