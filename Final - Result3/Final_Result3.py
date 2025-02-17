import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pretty_midi

#data preprocessing

DEFAULT_HYPERPARAMS = {
    'batch_size': 32,
    'block_size': 384,
    'max_iters': 1000,
    'eval_interval': 500,
    'learning_rate': 3e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_iters': 200,
    'n_embd': 192,
    'n_head': 4,
    'n_layer': 3,
    'dropout': 0.2,
    'early_stopping_patience': 500,
}

torch.manual_seed(1337)
random.seed(1337)

# Load your preprocessed melody data
with open('inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique tokens (vocab_size): {vocab_size}")
print(f"Tokens: {chars}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data augmentation

def pitch_shift(melody, semitones):
    shifted_melody = []
    for token in melody:
        if token == 'R':
            shifted_melody.append(token)
        elif token in chars:
            # If token has a sharp
            if len(token) > 1 and token[1] == '#':
                base = token[0]
                base_index = chars.index(base)
                shifted_index = (base_index + semitones) % 12
                new_token = chars[shifted_index] + '#'
                if new_token not in chars:
                    new_token = chars[shifted_index]
                shifted_melody.append(new_token)
            else:
                base = token
                base_index = chars.index(base)
                shifted_index = (base_index + semitones) % 12
                new_token = chars[shifted_index]
                shifted_melody.append(new_token)
        else:
            print(f"Warning: Skipping invalid token '{token}'")
    return shifted_melody

def augment_data(data_tensor):
    augmented_data = []
    data_str = decode(data_tensor.tolist())
    melodies = data_str.split('R')
    for melody in melodies:
        melody_tokens = list(melody)
        for semitone_shift in [-1, 1]:
            shifted_melody = pitch_shift(melody_tokens, semitone_shift)
            augmented_data.extend(shifted_melody + ['R'])
    return torch.tensor(encode(''.join(augmented_data)), dtype=torch.long)

augmented_train_data = augment_data(train_data)
augmented_val_data = augment_data(val_data)
train_data = augmented_train_data
val_data = augmented_val_data
print(f"After augmentation - Train data size: {len(train_data)}, Val data size: {len(val_data)}")

#loading the data

def get_batch(split, batch_size, block_size):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(DEFAULT_HYPERPARAMS['device']), y.to(DEFAULT_HYPERPARAMS['device'])
    return x, y

@torch.no_grad()
def estimate_loss(model, split, eval_iters, batch_size, block_size):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split, batch_size, block_size)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return float(losses.mean())  # Convert to float

#model

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
         # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
         # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=DEFAULT_HYPERPARAMS['device'])
        )
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -DEFAULT_HYPERPARAMS['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

#incorporating a baseline model for comprison


class MarkovChainBaseline:
    def __init__(self, data, vocab_size):
        self.vocab_size = vocab_size
        self.transitions = {}
        self.build_transitions(data)

    def build_transitions(self, data):
        for i in range(len(data) - 1):
            curr = data[i]
            nxt = data[i + 1]
            if curr not in self.transitions:
                self.transitions[curr] = {}
            if nxt not in self.transitions[curr]:
                self.transitions[curr][nxt] = 0
            self.transitions[curr][nxt] += 1
        for curr, nxts in self.transitions.items():
            total = sum(nxts.values())
            for tkn in nxts:
                nxts[tkn] /= total

    def generate(self, start_token, max_length):
        current_token = start_token
        generated = [current_token]
        for _ in range(max_length - 1):
            if current_token not in self.transitions:
                break
            next_tokens = list(self.transitions[current_token].keys())
            probabilities = list(self.transitions[current_token].values())
            chosen = random.choices(next_tokens, weights=probabilities, k=1)[0]
            generated.append(chosen)
            current_token = chosen
        return generated

#checking the comparison and creating plots 

def sanitize_config(config):
    items = []
    for k, v in config.items():
        # convert everything to string
        key_val = f"{k}-{v}"
        # remove special chars
        key_val = key_val.replace('.', '_').replace(':', '_')
        key_val = key_val.replace('{', '_').replace('}', '_').replace(',', '_')
        items.append(key_val)
    return "_".join(items)

def save_checkpoint(model, config, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=DEFAULT_HYPERPARAMS['block_size'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config

def plot_losses(history, config):
    plt.figure(figsize=(10,5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"Loss Curves for Config: {config}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    filename = f"loss_curve_{sanitize_config(config)}.png"
    plt.savefig(filename)
    plt.close()

hyperparams = [
    {'n_embd': 128, 'n_head': 2, 'n_layer': 2, 'dropout': 0.2},
    {'n_embd': 128, 'n_head': 4, 'n_layer': 3, 'dropout': 0.2},
    {'n_embd': 192, 'n_head': 2, 'n_layer': 2, 'dropout': 0.1},
    {'n_embd': 192, 'n_head': 4, 'n_layer': 2, 'dropout': 0.3},
]


subset_max_iters = 1000
subset_eval_interval = 200
subset_max_tokens = 100
log_file = "hyperparam_tuning_results.json"
results = []

if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        json.dump([], f)

# Global variables to track the best model across all configurations
global_best_val_loss = float('inf')  # Global best validation loss
global_best_config = None  # Best configuration globally
global_best_checkpoint_path = "global_best_model.pth"

def train_and_evaluate(config):
    global global_best_val_loss, global_best_config, global_best_checkpoint_path
    print(f"\nTraining with config: {config}")
    current_hyperparams = deepcopy(DEFAULT_HYPERPARAMS)
    current_hyperparams.update(config)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=current_hyperparams['n_embd'],
        n_head=current_hyperparams['n_head'],
        n_layer=current_hyperparams['n_layer'],
        block_size=current_hyperparams['block_size'],
        dropout=current_hyperparams['dropout']
    ).to(current_hyperparams['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=current_hyperparams['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=100, verbose=True)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    checkpoint_path = "best_model.pth"

    for iteration in range(subset_max_iters):
        xb, yb = get_batch('train', current_hyperparams['batch_size'], current_hyperparams['block_size'])
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # store float value
        history['train_loss'].append(float(loss.item()))

        # Evaluate at intervals
        if iteration % subset_eval_interval == 0 or iteration == subset_max_iters - 1:
            train_loss = estimate_loss(model, 'train',
                                       current_hyperparams['eval_iters'],
                                       current_hyperparams['batch_size'],
                                       current_hyperparams['block_size'])
            val_loss = estimate_loss(model, 'val',
                                     current_hyperparams['eval_iters'],
                                     current_hyperparams['batch_size'],
                                     current_hyperparams['block_size'])
            history['val_loss'].append(float(val_loss))

            print(f"Step {iteration}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                save_checkpoint(model, config, checkpoint_path)
            else:
                early_stopping_counter += subset_eval_interval

            if early_stopping_counter >= current_hyperparams['early_stopping_patience']:
                print("Early stopping triggered.")
                break

    # Generate sample melody
    context = torch.zeros((1, 1), dtype=torch.long, device=current_hyperparams['device'])
    generated_indices = model.generate(context, max_new_tokens=subset_max_tokens)[0].tolist()
    generated_melody = decode(generated_indices)

    # Convert everything to plain Python types
    result = {
        'config': {k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in config.items()},
        'best_val_loss': float(best_val_loss),
        'generated_melody': generated_melody,
        'checkpoint_path': str(checkpoint_path),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
        },
    }

    plot_losses(result['history'], result['config'])
    return result

for cfg in hyperparams:
    result = train_and_evaluate(cfg)
    results.append(result)
    # Dump the results as plain python objects (floats, lists, strings, etc.)
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

best_result = min(results, key=lambda x: x['best_val_loss'])
best_config = best_result['config']
print(f"\nBest Config: {best_config} with Val Loss: {best_result['best_val_loss']:.4f}")

#training

print("\nStarting final training with the best hyperparameters...")
final_hyperparams = deepcopy(DEFAULT_HYPERPARAMS)
# convert any floats that might've become strings back to numeric
for k, v in best_config.items():
    if isinstance(v, str) and v.isdigit():
        best_config[k] = int(v)
final_hyperparams.update(best_config)

final_model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=int(final_hyperparams['n_embd']),
    n_head=int(final_hyperparams['n_head']),
    n_layer=int(final_hyperparams['n_layer']),
    block_size=int(final_hyperparams['block_size']),
    dropout=float(final_hyperparams['dropout'])
).to(final_hyperparams['device'])

checkpoint_path = best_result['checkpoint_path']
final_model, loaded_config = load_checkpoint(checkpoint_path, final_hyperparams['device'])
print(f"Loaded best model from {checkpoint_path}")

optimizer = torch.optim.AdamW(final_model.parameters(), lr=final_hyperparams['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100, verbose=True
)

best_val_loss = float('inf')
early_stopping_counter = 0
final_history = {'train_loss': [], 'val_loss': []}

for i in range(final_hyperparams['max_iters']):
    xb, yb = get_batch('train', final_hyperparams['batch_size'], final_hyperparams['block_size'])
    logits, loss = final_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    final_history['train_loss'].append(float(loss.item()))

    if i % final_hyperparams['eval_interval'] == 0 or i == final_hyperparams['max_iters'] - 1:
        tr_loss = estimate_loss(final_model, 'train',
                                final_hyperparams['eval_iters'],
                                final_hyperparams['batch_size'],
                                final_hyperparams['block_size'])
        va_loss = estimate_loss(final_model, 'val',
                                final_hyperparams['eval_iters'],
                                final_hyperparams['batch_size'],
                                final_hyperparams['block_size'])
        final_history['val_loss'].append(float(va_loss))

        print(f"Step {i}: Train Loss {tr_loss:.4f}, Val Loss {va_loss:.4f}")

        scheduler.step(va_loss)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            early_stopping_counter = 0
            final_checkpoint_path = "final_best_model.pth"
            save_checkpoint(final_model, best_config, final_checkpoint_path)
        else:
            early_stopping_counter += final_hyperparams['eval_interval']

        if early_stopping_counter >= final_hyperparams['early_stopping_patience']:
            print("Early stopping triggered.")
            break

plot_losses(final_history, best_config)

#to evaluate and validate the performance

def melody_to_midi(melody_str, output_path='generated_melody.mid', tempo=120):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    current_time = 0
    for token in melody_str:
        if token == 'R':
            current_time += 0.5
        else:
            base_pitch = 60
            note_index = chars.index(token)
            pitch = base_pitch + note_index
            note = pretty_midi.Note(velocity=100, pitch=pitch,
                                    start=current_time, end=current_time + 0.5)
            instrument.notes.append(note)
            current_time += 0.5
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"MIDI file saved to {output_path}")

final_model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=final_hyperparams['device'])
generated_indices = final_model.generate(context, max_new_tokens=500)[0].tolist()
generated_melody = decode(generated_indices)
print("Final Generated Melody:")
print(generated_melody)
melody_to_midi(generated_melody, 'final_generated_melody.mid')

def calculate_perplexity(loss):
    return float(torch.exp(torch.tensor(loss)))

train_loss = estimate_loss(final_model, 'train',
                           final_hyperparams['eval_iters'],
                           final_hyperparams['batch_size'],
                           final_hyperparams['block_size'])
val_loss = estimate_loss(final_model, 'val',
                         final_hyperparams['eval_iters'],
                         final_hyperparams['batch_size'],
                         final_hyperparams['block_size'])
train_perplexity = calculate_perplexity(train_loss)
val_perplexity = calculate_perplexity(val_loss)

print(f"Final Train Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.4f}")
print(f"Final Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")

baseline = MarkovChainBaseline(train_data.tolist(), vocab_size)
start_token = random.choice(train_data.tolist())
baseline_generated = baseline.generate(start_token, max_length=500)
baseline_melody = decode(baseline_generated)
print("Baseline Generated Melody:")
print(baseline_melody)
melody_to_midi(baseline_melody, 'baseline_generated_melody.mid')

def baseline_perplexity(melody, transitions):
    log_prob = 0.0
    count = 0
    for i in range(len(melody) - 1):
        current = melody[i]
        nxt = melody[i + 1]
        if current in transitions and nxt in transitions[current]:
            prob = transitions[current][nxt]
            log_prob += -torch.log(torch.tensor(prob))
        else:
            log_prob += -torch.log(torch.tensor(1e-6))
        count += 1
    if count == 0:
        return float('inf')
    avg_log_prob = log_prob / count
    return float(torch.exp(avg_log_prob))

baseline_ppl = baseline_perplexity(baseline_generated, baseline.transitions)
print(f"Baseline Perplexity: {baseline_ppl:.4f}")

print("\nComparison:")
print(f"GPT Model Validation Perplexity: {val_perplexity:.4f}")
print(f"Baseline Perplexity: {baseline_ppl:.4f}")

final_results = {
    'final_model': {
        'config': best_config,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_perplexity': train_perplexity,
        'val_perplexity': val_perplexity,
        'generated_melody': generated_melody,
    },
    'baseline': {
        'generated_melody': baseline_melody,
        'perplexity': baseline_ppl,
    }
}

with open('final_results.json', 'w') as f:
    json.dump(final_results, f, indent=4)

with open(log_file, 'a') as f:
    json.dump({'final_results': final_results}, f, indent=4)

print("Final results saved to 'final_results.json'.")

plot_losses(final_history, best_config)
plt.title("Final Training Loss Curve")
plt.savefig("final_training_loss.png")

