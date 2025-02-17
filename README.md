# MelodyGPT: AI-Powered Melody Generation with Transformers
A Machine Learning project that repurposes GPT for melody generation using MIDI datasets, augmentation techniques, and hyperparameter tuning to improve music generation quality.

 1. Overview
This project builds a melody-generating model using a Transformer-based architecture (GPT). The dataset consists of simplified MIDI melodies converted into note sequences. The model is trained on an augmented dataset that includes:
- Pitch Transpositions (±1, ±2 semitones)
- Random Rest Insertions (introducing rhythmic variations)
- Duplicate Removal (reducing redundancy and improving model diversity)

 2. Project Structure

📂 Final - Result3/
│── augmentMidiTranslations.py      # Augments MIDI dataset (transposition, rest insertion, duplicate removal)
│── baseline_generated_melody.mid   # Baseline melody from Markov Model
│── checkgpu.py                      # Check system GPU availability
│── extractMelodies.py               # Extract melodies from MIDI dataset
│── final_best_model.pth             # Best-trained GPT model
│── final_generated_melody.mid       # GPT-generated melody
│── Final_hyperparam_tuning_results.json  # Hyperparameter tuning results
│── Final_Result3.py                 # Main training script (modified from GPT)
│── final_results.json                # Results for different runs
│── final_training_loss.png           # Loss curve visualization
│── gpt.py                            # Original GPT-based melody generation script
│── inputMelodies.txt                 # Simplified melodies before augmentation
│── inputMelodiesAugmented.txt        # Final dataset with enhancements
│── melodyPlay.py                     # Play generated melodies
│── midi2text.py                      # Converts MIDI files into text representations
│── README.md                         # This file
│── Result3_loss_curve_*.png          # Loss curves for different hyperparameter settings


 3. Installation
Prerequisites
- Python 3.8+
- PyTorch (GPU recommended)
- Required dependencies:
  
  pip install numpy torch torchvision torchaudio matplotlib pydub pretty_midi simpleaudio
  
  If simpleaudio fails to install on Windows, use:
  
  pip install pipwin
  pipwin install simpleaudio
  
  On Linux/macOS, ensure you have development tools:
  
  sudo apt install build-essential python3-dev
  

 4. Dataset Processing & Augmentation
The project processes MIDI melodies into text representations and applies augmentation techniques:
- `midi2text.py` → Converts MIDI files into a text format (notes & rests).
- `augmentMidiTranslations.py` → Applies ±1 and ±2 semitone shifts, inserts random rests, and removes duplicate sequences.
- The final dataset is stored in `inputMelodiesAugmented.txt`.

 5. Model Training
Run:

python Final_Result3.py

This will:
1. Load the dataset and apply preprocessing.
2. Train a GPT-based model using the best-tuned hyperparameters.
3. Evaluate performance using cross-entropy loss and perplexity.
4. Generate new melodies stored as `.mid` files.

 6. Baseline vs. GPT Model
| Metric         | GPT Model (Final) | Baseline Markov Model |
|---------------|----------------|-----------------|
| Validation Perplexity | 2.08 | 4.90 |
| Train Perplexity | 2.10 | 4.90 |
| Melody Structure | Less repetitive, diverse phrasing | Random transitions |

 7. Hyperparameter Tuning
Tested different configurations:
- `n_embd`: {128, 192}
- `n_layer`: {2, 3}
- `n_head`: {2, 4}
- `dropout`: {0.1, 0.2, 0.3}

The best configuration found:

{'n_embd': 192, 'n_head': 2, 'n_layer': 2, 'dropout': 0.1}

Resulting in the lowest validation perplexity (2.08).

 8. Generating Melodies
Once trained, the model can generate melodies:

python melodyPlay.py

- Final Model Output: `final_generated_melody.mid`
- Baseline Output: `baseline_generated_melody.mid`

 9. Evaluation Metrics
- Cross-Entropy Loss: Measures how well the model predicts the next note.
- Perplexity: Measures uncertainty in predictions (lower is better).
- ROC-AUC Curve: Used for classification evaluation.

 10. Conclusion
This project successfully adapts GPT for melody generation, proving that transformer models can learn meaningful musical patterns. By incorporating data augmentation, hyperparameter tuning, and a comparative baseline, it demonstrates how deep learning can be used for creative AI applications.
