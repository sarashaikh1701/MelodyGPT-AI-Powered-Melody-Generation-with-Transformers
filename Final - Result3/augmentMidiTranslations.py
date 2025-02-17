# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:06:10 2024

@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

def translate_notes(line, shift):
    NOTES = ['C','c','D','d','E','F','f','G','g','A','a','B']
    tokens = line.strip().split()
    new_tokens = []
    for tok in tokens:
        if tok in NOTES:
            i = NOTES.index(tok)
            i_new = (i + shift) % len(NOTES)
            new_tokens.append(NOTES[i_new])
        elif tok == 'R':
            new_tokens.append('R')
        else:
            # If there's anything else, keep it
            new_tokens.append(tok)
    return ' '.join(new_tokens)

if __name__ == "__main__":
    # read the base lines
    with open('inputMelodies.txt','r') as f:
        base_lines = [ln.strip() for ln in f if ln.strip()]

    # define new transpositions
    shifts = [-2, -1, 1, 2]

    all_augmented = []
    for line in base_lines:
        for sh in shifts:
            new_line = translate_notes(line, sh)
            if new_line.strip():
                all_augmented.append(new_line)

    # remove duplicates
    unique_set = set()
    final_lines = []
    for ln in all_augmented:
        if ln not in unique_set:
            unique_set.add(ln)
            final_lines.append(ln)

    # write to file
    with open('inputMelodiesAugmented.txt','w') as f:
        for ln in final_lines:
            f.write(ln + "\n")

    print("Augmented lines with ±2 semitone shifts, duplicates removed, saved to inputMelodiesAugmented.txt")
