from midiutil import MIDIFile
import os

# Define root notes and intervals
root_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
note_to_midi = {
    'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
    'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71
}
major_intervals = [0, 4, 7]
minor_intervals = [0, 3, 7]

output_dir = "midi_chords"
os.makedirs(output_dir, exist_ok=True)

for root in root_notes:
    for chord_type, intervals in [('major', major_intervals), ('minor', minor_intervals)]:
        midi = MIDIFile(1)
        track = 0
        time = 0
        midi.addTrackName(track, time, f"{root} {chord_type}")
        midi.addTempo(track, time, 120)

        root_midi = note_to_midi[root]
        for interval in intervals:
            midi.addNote(track, channel=0, pitch=root_midi + interval, time=0, duration=2, volume=100)

        file_name = f"{root}_{chord_type}.mid"
        with open(os.path.join(output_dir, file_name), "wb") as output_file:
            midi.writeFile(output_file)

print("✅ 24 chord MIDI files generated in the 'midi_chords' folder.")
