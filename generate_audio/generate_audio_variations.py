import os
import mido
import subprocess
from mido import MidiFile, MidiTrack, Message

# Config
MIDI_DIR = "midi_chords"
WAV_DIR = "wav_dataset"
SOUNDFONT = r"C:\Soundfonts\GeneralUser-GS.sf2"
PITCH_SHIFTS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
VELOCITY_MULTIPLIERS = [0.8, 1.0, 1.2]

os.makedirs(WAV_DIR, exist_ok=True)

def modify_midi(input_path, pitch_shift, velocity_multiplier):
    midi = MidiFile(input_path)
    new_midi = MidiFile()
    
    for track in midi.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type in ["note_on", "note_off"]:
                new_note = msg.note  # don't shift MIDI note numbers
                new_velocity = int(min(127, max(1, msg.velocity * velocity_multiplier)))
                new_msg = msg.copy(note=new_note, velocity=new_velocity)
                new_track.append(new_msg)
            else:
                new_track.append(msg)
        new_midi.tracks.append(new_track)

    temp_path = input_path.replace(".mid", f"_temp.mid")
    new_midi.save(temp_path)
    return temp_path

def pitch_shift_wav(input_wav, output_wav, n_steps):
    import librosa
    import soundfile as sf

    y, sr = librosa.load(input_wav, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(output_wav, y_shifted, sr)

def render_midi_to_wav(midi_path, wav_path):
    command = [
        "fluidsynth",
        "-ni",
        "-T", "wav",
        "-F", wav_path,
        SOUNDFONT,
        midi_path
    ]
    subprocess.run(command, check=True)

# Main loop
for file in os.listdir(MIDI_DIR):
    if file.endswith(".mid"):
        chord_name = file.replace(".mid", "")
        input_midi = os.path.join(MIDI_DIR, file)

        for pitch_shift in PITCH_SHIFTS:
            for velocity_multiplier in VELOCITY_MULTIPLIERS:
                # Modify MIDI (velocity only)
                temp_midi = modify_midi(input_midi, pitch_shift, velocity_multiplier)

                # Temp WAV before pitch shifting
                temp_wav = os.path.join(WAV_DIR, "temp.wav")
                render_midi_to_wav(temp_midi, temp_wav)

                # Final output WAV path
                pitch_tag = f"{'n' if pitch_shift < 0 else 'p'}{abs(pitch_shift):.1f}".replace(".", "")
                velocity_tag = f"v{int(velocity_multiplier*100)}"
                output_name = f"{chord_name}_ps{pitch_tag}_{velocity_tag}.wav"
                output_wav = os.path.join(WAV_DIR, output_name)

                # Apply pitch shift to audio
                pitch_shift_wav(temp_wav, output_wav, pitch_shift)

                # Cleanup
                os.remove(temp_midi)
                os.remove(temp_wav)

print("✅ All variations generated.")
