
# -*- coding: utf-8 -*-
"""
@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

import os
from mido import MidiFile, MidiTrack, Message
import random

REST_PROB = 0.25     # Probability of inserting a random rest
REST_INTERVAL = 8    # Insert the rest roughly every 8 notes if triggered

# Map MIDI note to single-octave note names
MIDI_NOTE_TO_NAME = {
    0: 'C', 1: 'c', 2: 'D', 3: 'd', 4: 'E', 5: 'F', 
    6: 'f', 7: 'G', 8: 'g', 9: 'A', 10: 'a', 11: 'B'
}

def randomly_insert_rests(note_tokens, rest_prob=0.25, interval=8):
    """
    After every 'interval' notes, we may insert a 'R' token with probability rest_prob.
    """
    new_tokens = []
    count_since_rest = 0
    for token in note_tokens:
        new_tokens.append(token)
        if token != 'R':
            count_since_rest += 1
        # Attempt insertion after a certain interval
        if count_since_rest >= interval:
            if random.random() < rest_prob:
                new_tokens.append('R')
            count_since_rest = 0
    return new_tokens

def midi_to_text_sequence(midi_path):
    midi = MidiFile(midi_path)
    sequence = []
    last_note_time = 0

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note = MIDI_NOTE_TO_NAME.get(msg.note % 12, '')
                if note:
                    # simple approach: if msg.time>0, treat them as rests
                    rest_duration = msg.time // 480  # each 480 => 1 beat
                    if rest_duration > 0:
                        sequence.extend(['R'] * rest_duration)
                    sequence.append(note)
    # Some lines may have consecutive 'R's
    # We'll only remove direct duplicates
    # Then do random rest insertion
    final_seq = []
    for i, tok in enumerate(sequence):
        # skip if consecutive 'R R'
        if tok == 'R' and i > 0 and sequence[i-1] == 'R':
            continue
        final_seq.append(tok)

    # Convert to minor rest insertion
    final_seq = randomly_insert_rests(final_seq, REST_PROB, REST_INTERVAL)

    # Join them with space
    return ' '.join(final_seq)

def text_sequence_to_midi(sequence, output_path):
    # unchanged code for reversing the process if you want
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    tokens = sequence.strip().split()

    # We assume each note is from MIDI_NOTE_TO_NAME reversed
    name_to_midi = {v:k for k,v in MIDI_NOTE_TO_NAME.items()}

    # We'll define a base note offset
    base_offset = 60  # Middle C is 60
    time_accum = 0

    for tok in tokens:
        if tok == 'R':
            # add rests as time
            time_accum += 480
        else:
            # get midi note from name
            if tok in name_to_midi:
                note_val = name_to_midi[tok] + base_offset
                track.append(Message('note_on', note=note_val, velocity=64, time=time_accum))
                track.append(Message('note_off', note=note_val, velocity=64, time=480))
                time_accum = 0
    midi.save(output_path)

# Main script:
midi_dir = 'musicDatasetOriginal'
output_dir = 'musicDatasetSimplified'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

text_sequences = []

for file_name in os.listdir(midi_dir):
    if file_name.endswith('.mid'):
        midi_path = os.path.join(midi_dir, file_name)
        text_sequence = midi_to_text_sequence(midi_path)
        if text_sequence.strip():
            text_sequences.append(text_sequence)
        else:
            print(f"No notes found in {file_name}")

# Write lines
with open("inputMelodies.txt", "w") as file:
    for seq in text_sequences:
        file.write(seq + "\n")

# Also optionally convert back to midi for checking
for i, seq in enumerate(text_sequences):
    output_midi = os.path.join(output_dir, f"output_midi_{i+1}.mid")
    text_sequence_to_midi(seq, output_midi)

print("Text sequences with minor rest insertion written to inputMelodies.txt")
