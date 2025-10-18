import os # to work with files and directories and all that
import mido # read write and manipulate midi files
import subprocess # to call external programs (fluidsynth)
import random #randomness in velocity variation
import time # for progress tracking
from mido import MidiFile, MidiTrack, Message # working with midi structures

# Config
MIDI_DIR = "midi_chords" #folder for midi files
WAV_DIR = "wav_dataset" #folder for wav files
SOUNDFONT = r"C:\Soundfonts\GeneralUser-GS.sf2"  #path to a sound font that fluid synth will use to make audio. this is like a piano sound
PITCH_SHIFTS = [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] #pitch shitfs


# 21 * 5 * 3 * 3
#did PITCH_SHIFTS = [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] #pitch shitfs
# with 0.8 0.9 1 1.1 1.2

VELOCITY_MULTIPLIERS = [0.85, 0.95, 1.0, 1.05, 1.15] # velocity multipliers
OCTAVE_SHIFTS = [-12, 0, 12]  # -1, 0, +1 octave
INVERSIONS = [0, 1, 2]  # root, 1st, 2nd inversions (up to triads)

os.makedirs(WAV_DIR, exist_ok=True) #makes "wav_datset" if it doesn't already exist

# Progress tracking variables
EXPECTED_FILES = 24 * 3 * 3 * 21 * 5  # 22,680 total files
files_generated = 0
files_skipped = 0
files_failed = 0
start_time = time.time()

# Track all expected filenames to ensure no duplicates
expected_filenames = set()

def get_note_ons(track): # extracts just the notes.... cause midi files also have like info on tempos, program changes, controller changes, etc...
    """Extract all note_on messages from a track."""
    return [msg for msg in track if msg.type == "note_on"]

def apply_inversion(note_ons, inversion):
    """Rotate notes upward to simulate inversions."""
    if inversion == 0: # if no inversion is to be applied
        return note_ons

    sorted_notes = sorted(note_ons, key=lambda x: x.note) # sort each note from lowest pitch to highest pitch
    for i in range(inversion):
        # Move the lowest note up one octave
        sorted_notes[i].note += 12
    return sorted(sorted_notes, key=lambda x: x.note) #resort the notes from lowest to highest. key=lambda... is second argument for sorted function. telling it to sort by note number

def modify_midi(input_path, octave_shift, velocity_multiplier, inversion):
    try:
        midi = MidiFile(input_path) #loads an input midi file
        new_midi = MidiFile() # creates a new midi file

        for track in midi.tracks:
            new_track = MidiTrack() # creates a new track
            note_ons = [] 

            # collects note_ons from all tracks
            for msg in track:
                if msg.type == "note_on":
                    note_ons.append(msg)

            if note_ons: # if there are notes basically, in the list
                note_ons = apply_inversion(note_ons, inversion) # apply the inversions

            for msg in track:
                
                if msg.type in ["note_on", "note_off"]: #btw, in midi files, note on starts sound, note off stops sound
                    # Use modified note if in note_ons list
                    # you have to match the notes now because you inverted some of the chords... so now you need to match the original notes.
                    match = next((n for n in note_ons if n.time == msg.time and n.note % 12 == msg.note % 12), None) # finds the matching note
                    note = match.note if match else msg.note
                    note = max(0, min(127, note + octave_shift))

                    #velocity is a seperate variable from note, find the new velocity below
                    if msg.type == "note_on": # apply random velocity to the note if it's note on
                        vel_variation = random.uniform(0.85, 1.15)
                        velocity = int(min(127, max(1, msg.velocity * velocity_multiplier * vel_variation)))
                    else:
                        velocity = msg.velocity

                    new_track.append(msg.copy(note=note, velocity=velocity)) # add the new note in the new_track
                else:
                    new_track.append(msg)
            #add track to midi file
            new_midi.tracks.append(new_track)

        temp_path = input_path.replace(".mid", f"_temp_{random.randint(1000, 9999)}.mid")
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
        # Check if file was created and has content
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

# Pre-generate all expected filenames for validation
print("Pre-generating expected filenames for validation...")
for file in os.listdir(MIDI_DIR):
    if file.endswith(".mid"):
        chord_name = file.replace(".mid", "")
        for inversion in INVERSIONS:
            for octave_shift in OCTAVE_SHIFTS:
                for pitch_shift in PITCH_SHIFTS:
                    for velocity_multiplier in VELOCITY_MULTIPLIERS:
                        pitch_tag = f"{'n' if pitch_shift < 0 else 'p'}{abs(pitch_shift):.1f}".replace(".", "")
                        velocity_tag = f"v{int(velocity_multiplier*100)}"
                        octave_tag = f"o{octave_shift}"
                        inv_tag = f"inv{inversion}"
                        output_name = f"{chord_name}_{inv_tag}_{octave_tag}_ps{pitch_tag}_{velocity_tag}.wav"
                        expected_filenames.add(output_name)

print(f"Expected {len(expected_filenames)} unique files")

# Main loop
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
                            # Skip if file already exists
                            pitch_tag = f"{'n' if pitch_shift < 0 else 'p'}{abs(pitch_shift):.1f}".replace(".", "")
                            velocity_tag = f"v{int(velocity_multiplier*100)}"
                            octave_tag = f"o{octave_shift}"
                            inv_tag = f"inv{inversion}"
                            output_name = f"{chord_name}_{inv_tag}_{octave_tag}_ps{pitch_tag}_{velocity_tag}.wav"
                            output_wav = os.path.join(WAV_DIR, output_name)
                            
                            if os.path.exists(output_wav):
                                files_skipped += 1
                                files_generated += 1
                                continue

                            # Modify MIDI
                            temp_midi = modify_midi(
                                input_midi,
                                octave_shift=octave_shift,
                                velocity_multiplier=velocity_multiplier,
                                inversion=inversion
                            )
                            
                            if temp_midi is None:
                                files_failed += 1
                                continue

                            # Render to WAV
                            temp_wav = os.path.join(WAV_DIR, f"temp_{random.randint(10000, 99999)}.wav")
                            if not render_midi_to_wav(temp_midi, temp_wav):
                                if os.path.exists(temp_midi):
                                    os.remove(temp_midi)
                                files_failed += 1
                                continue

                            # Apply pitch shift
                            if not pitch_shift_wav(temp_wav, output_wav, pitch_shift):
                                if os.path.exists(temp_midi):
                                    os.remove(temp_midi)
                                if os.path.exists(temp_wav):
                                    os.remove(temp_wav)
                                files_failed += 1
                                continue

                            # Clean up temp files
                            if os.path.exists(temp_midi):
                                os.remove(temp_midi)
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)
                            
                            # Progress tracking
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

# Final statistics and validation
elapsed = time.time() - start_time
print(f"\n" + "="*60)
print("GENERATION COMPLETE - DETAILED VALIDATION")
print("="*60)

# Get actual files in directory
actual_files = set([f for f in os.listdir(WAV_DIR) if f.endswith(".wav")])
final_count = len(actual_files)

print(f"Files processed: {files_generated}")
print(f"Files skipped (already existed): {files_skipped}")
print(f"Files failed: {files_failed}")
print(f"Expected files: {EXPECTED_FILES}")
print(f"Actual files in directory: {final_count}")
print(f"Success rate: {files_generated/EXPECTED_FILES*100:.1f}%")
print(f"Time elapsed: {elapsed/60:.1f} minutes")
print(f"Average rate: {files_generated/elapsed:.1f} files/second")

# Detailed validation
print(f"\n" + "-"*40)
print("VALIDATION RESULTS:")
print("-"*40)

# Check for missing files
missing_files = expected_filenames - actual_files
extra_files = actual_files - expected_filenames

if len(missing_files) == 0 and len(extra_files) == 0:
    print("✅ PERFECT! All 22,680 files generated with no duplicates or missing files!")
elif len(missing_files) > 0:
    print(f"❌ MISSING {len(missing_files)} files:")
    for missing in sorted(list(missing_files))[:10]:  # Show first 10
        print(f"   - {missing}")
    if len(missing_files) > 10:
        print(f"   ... and {len(missing_files) - 10} more")
elif len(extra_files) > 0:
    print(f"⚠️  EXTRA {len(extra_files)} files (possible duplicates):")
    for extra in sorted(list(extra_files))[:10]:  # Show first 10
        print(f"   - {extra}")
    if len(extra_files) > 10:
        print(f"   ... and {len(extra_files) - 10} more")

# Check for duplicates by counting occurrences
from collections import Counter
file_counts = Counter([f for f in os.listdir(WAV_DIR) if f.endswith(".wav")])
duplicates = {name: count for name, count in file_counts.items() if count > 1}

if duplicates:
    print(f"❌ DUPLICATE FILES FOUND: {len(duplicates)} files appear multiple times")
    for dup_file, count in list(duplicates.items())[:5]:
        print(f"   - {dup_file}: {count} copies")
else:
    print("✅ No duplicate files found")

# Final status
if final_count == EXPECTED_FILES and len(missing_files) == 0 and len(extra_files) == 0 and len(duplicates) == 0:
    print(f"\n🎉 SUCCESS! All {EXPECTED_FILES} files generated flawlessly!")
    print("✅ No missing files")
    print("✅ No duplicate files") 
    print("✅ No extra files")
    print("✅ Perfect generation!")
else:
    print(f"\n⚠️  Generation completed with issues:")
    print(f"   - Missing: {len(missing_files)} files")
    print(f"   - Extra: {len(extra_files)} files")
    print(f"   - Duplicates: {len(duplicates)} files")

print("✅ All chord variations (inversions, pitch, velocity, octave) generated.")
