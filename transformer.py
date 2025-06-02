import kagglehub
import os
import glob
import pretty_midi
import librosa
from basic_pitch.inference import Model, predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from musicPlayer import MusicPlayer
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset
from torch.utils.data import Dataset
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
from miditok import REMI, TokenizerConfig, TokSequence
import numpy as np
from tqdm import tqdm
import time


# # Add this at the top of your file with other imports
# sys.setrecursionlimit(100000)  # Increase recursion limit

# # Download latest version
# path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

# print("Path to dataset files:", path)


# basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

# genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# # Add after imports
# # Special tokens
# PAD = 0  # Padding token
# BOS = 1  # Beginning of sequence token
# EOS = 2  # End of sequence token

# # Create genre mapping
# GENRE_TO_ID = {genre: idx for idx, genre in enumerate(genres)}
# ID_TO_GENRE = {idx: genre for genre, idx in GENRE_TO_ID.items()}

# def convert_wav_to_midi(wav_path, output_dir):
#     # Run prediction
#     try:
#         model_output, midi_data, note_events = predict(
#             wav_path,
#             basic_pitch_model,
#         )

#         # Save MIDI
#         base_name = os.path.splitext(os.path.basename(wav_path))[0]
#         midi_path = os.path.join(output_dir, f"{base_name}.mid")
#         midi_data.write(midi_path)
#         print(f"[] Saved: {midi_path}")
#     except Exception as e:
#         print(f"Error: {e}")

# def batch_convert(folder_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
#     print(wav_files)
#     if not wav_files:
#         print("No .wav files found.")
#         return

#     for wav_file in wav_files:
#         convert_wav_to_midi(wav_file, output_dir)
# # Example usage
# input_folder = "Data/genres_original"
# output_folder = "Data/midi_output"

# genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
# if not os.path.exists(output_folder):
#     for genre in genres:
#         temp_input_folder = input_folder + "/" + genre
#         temp_output_folder = output_folder + "/" + genre
#         print(temp_input_folder)
#         print(temp_output_folder)
#         batch_convert(temp_input_folder, temp_output_folder)
# else:
#     print("Output folder already exists.")

# # musicPlayer = MusicPlayer("gtzan-dataset-music-genre-classification/versions/1/Data/midi_output/blues/blues.00000.mid")
# # musicPlayer.play_music()



# config = TokenizerConfig()
# remi_tokenizer = REMI(config)
# dataset_path = "dataset.txt"

# a = 0
# def tokenize_midi_folder(midi_folder, output_file):
#     genre = midi_folder.split("/")[-1]
#     print(f'Tokenizing {genre}...')
#     all_tokens = []

#     for midi_path in os.listdir(midi_folder):
#         if not midi_path.endswith(".mid"):
#             continue
#         midi = MidiFile(os.path.join(midi_folder, midi_path))
#         # Add genre token at the start
#         genre_token = f"<|{genre}|>"
#         tokens = remi_tokenizer(midi) # List[TokSequence]
#         # print(f'tokens: {tokens}')
#         # print(f'tokens[0].ids: {tokens[0].ids}')
#         # Convert REMI tokens to their integer values
#         all_sequences = [token for token in tokens[0].ids]
#         # Convert tokens to strings and add special tokens
#         token_str = f"{genre_token} {' '.join(map(str, all_sequences))}"  # Add separator after first sequence
#         all_tokens.append(token_str)

#     # Save token sequences as dataset
#     with open(output_file, "a", encoding='utf-8') as f:
#         for sequence in all_tokens:
#             f.write(sequence + "\n")

# if not os.path.exists(dataset_path):
#     for genre in genres:
#         tokenize_midi_folder(output_folder + "/" + genre, dataset_path)
# else:
#     print(f"Dataset already exists in {dataset_path}")


genres = ['Ambient', 'Blues', 'Children', 'Classical', 'Country', 'Electronic', 'Folk', \
    'Jazz', 'Latin', 'Pop', 'Rap', 'Reggae', 'Religious', 'Rock', 'Soundtrack', 'Unknown', 'World']

PAD = 0  # Padding token
BOS = 1  # Beginning of sequence token
EOS = 2  # End of sequence token

# Create genre mapping
GENRE_TO_ID = {genre: idx for idx, genre in enumerate(genres)}
ID_TO_GENRE = {idx: genre for genre, idx in GENRE_TO_ID.items()}
remi_tokenizer = REMI(TokenizerConfig())

new_dataset_path = "new_dataset.txt"

def tokenize_midi_file(midi_file, genre):
    all_tokens = []

    try:
        # Load and preprocess MIDI file
        midi = MidiFile(midi_file)

        # Add genre token at the start
        genre_token = f"<|{genre}|>"
        
        # Convert to tokens
        tokens = remi_tokenizer(midi)
        
        if not tokens or not tokens[0].ids:
            print(f"Warning: No tokens generated for {midi_file}")
            return
            
        # Convert REMI tokens to their integer values
        all_sequences = [token for token in tokens[0].ids]
        
        # Convert tokens to strings and add special tokens
        token_str = f"{genre_token} {' '.join(map(str, all_sequences))}"
        all_tokens.append(token_str)

        # Save token sequences to dataset
        with open(new_dataset_path, "a", encoding='utf-8') as f:
            for sequence in all_tokens:
                f.write(sequence + "\n")
                
    except Exception as e:
        print(f"Error processing {midi_file}: {str(e)}")
        print("Skipping this file...")



midi_file_folder = "adl-piano-midi"
valid_files = 0
all_files = 0
if not os.path.exists(new_dataset_path):
    for genre in genres:
        genre_path = os.path.join(midi_file_folder, genre)
        if not os.path.isdir(genre_path):
            continue
            
        for music_style in os.listdir(genre_path):
            style_path = os.path.join(genre_path, music_style)
            if not os.path.isdir(style_path):
                continue
                
            for artist_folder in os.listdir(style_path):
                artist_path = os.path.join(style_path, artist_folder)
                if not os.path.isdir(artist_path):
                    continue
                    
                for midi_file in os.listdir(artist_path):
                    if not midi_file.endswith('.mid'):
                        continue
                        
                    midi_path = os.path.join(artist_path, midi_file)
                    try:
                        all_files += 1
                        tokenize_midi_file(midi_path, genre)
                        valid_files += 1
                    except Exception as e:
                        print(f"Error processing {midi_path}: {str(e)}")
    print(f"Valid files: {valid_files} out of {all_files}")
else:
    print(f"Dataset already exists in {new_dataset_path}")



# Create custom dataset class
class MIDIDataset(Dataset):
    def __init__(self, txt_file, genre_map, max_seq_len=512):
        self.samples = []
        self.genre_map = genre_map
        self.max_seq_len = max_seq_len
        
        print(f"Loading dataset from {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    # Extract genre from the format "<|genre|>"
                    genre_token = parts[0]
                    genre = genre_token.strip('<|>')
                    
                    if genre not in genre_map:
                        print(f"Warning: Unknown genre '{genre}' in line {line_num}")
                        continue
                    
                    # Convert MIDI tokens to integers
                    try:
                        midi_tokens = [int(token) for token in parts[1:]]
                    except ValueError:
                        print(f"Warning: Invalid token in line {line_num}")
                        continue
                    
                    # Truncate if too long
                    if len(midi_tokens) > max_seq_len - 2:  # -2 for BOS and EOS
                        midi_tokens = midi_tokens[:max_seq_len - 2]
                    
                    # Add special tokens
                    midi_tokens = [BOS] + midi_tokens + [EOS]
                    
                    # Pad sequence
                    if len(midi_tokens) < max_seq_len:
                        midi_tokens += [PAD] * (max_seq_len - len(midi_tokens))
                    
                    genre_id = genre_map[genre]
                    self.samples.append((genre_id, midi_tokens))
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        print(f"Loaded {len(self.samples)} samples")
        if len(self.samples) == 0:
            raise ValueError("No valid samples found in the dataset!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        genre_id, midi_tokens = self.samples[idx]
        # Return genre_id as a single integer, not wrapped in extra dimensions
        enc_input = torch.tensor(genre_id, dtype=torch.long)
        dec_input = torch.tensor(midi_tokens[:-1], dtype=torch.long)
        dec_target = torch.tensor(midi_tokens[1:], dtype=torch.long)
        return enc_input, dec_input, dec_target



class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size  # Store vocab_size for loading
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Genre embedding (smaller dimension since it's a simpler feature)
        self.genre_emb = nn.Embedding(len(genres), d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)

    def save_model(self, path):
        """Save the model state dict and configuration"""
        model_state = {
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_len': self.max_len
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path, device='cuda'):
        """Load a saved model"""
        model_state = torch.load(path, map_location=device)
        model = cls(
            vocab_size=model_state['vocab_size'],
            d_model=model_state['d_model'],
            max_len=model_state['max_len']
        ).to(device)
        model.load_state_dict(model_state['state_dict'])
        print(f"Model loaded from {path}")
        return model

    def create_padding_mask(self, seq):
        return seq == PAD
        
    def forward(self, genre_input, dec_input):
        """
        Args:
            genre_input: Genre ID tensor of shape [batch_size]
            dec_input: Decoder input tensor of shape [batch_size, seq_len]
        """
        batch_size = dec_input.size(0)
        seq_len = dec_input.size(1)
        device = dec_input.device
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=device)
        
        # Encoder (genre) processing
        genre_emb = self.genre_emb(genre_input)  # [batch_size, d_model]
        # Expand genre embedding to match sequence length
        genre_emb = genre_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        
        # Decoder processing
        dec_tok_emb = self.token_emb(dec_input)  # [batch_size, seq_len, d_model]
        dec_pos_emb = self.pos_emb(positions)  # [seq_len, d_model]
        dec_pos_emb = dec_pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]
        dec_emb = dec_tok_emb + dec_pos_emb
        
        # Create masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        tgt_mask = tgt_mask.to(torch.bool)  # Convert to boolean
        padding_mask = self.create_padding_mask(dec_input).to(device) if hasattr(self, 'create_padding_mask') else None
        if padding_mask is not None:
            padding_mask = padding_mask.to(torch.bool)  # Convert to boolean
        
        # Forward pass through transformer
        out = self.transformer(
            src=genre_emb,  # [batch_size, seq_len, d_model]
            tgt=dec_emb,    # [batch_size, seq_len, d_model]
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )
        
        return self.output_layer(out)  # [batch_size, seq_len, vocab_size]

    def generate(self, genre_token, max_length=500, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Convert genre token to tensor
            genre = genre_token.strip('<|>')
            genre_id = torch.tensor([GENRE_TO_ID[genre]], device=device)
            
            # Start with BOS token
            current_sequence = torch.tensor([[BOS]], dtype=torch.long, device=device)
            
            # Generate one token at a time
            for _ in range(max_length):
                # Get model predictions
                logits = self.forward(genre_id, current_sequence)
                
                # Get the last token's predictions
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we generate EOS
                if next_token.item() == EOS:
                    break
                    
                # Append to sequence
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
                
                # Stop if we reach max length
                if current_sequence.size(1) >= max_length:
                    break
            
            return current_sequence



print("Analyzing REMI tokenizer vocabulary...")
valid_tokens = set()
for token_id in range(1000):  # Check first 1000 potential tokens
    try:
        remi_tokenizer[token_id]
        valid_tokens.add(token_id)
    except:
        continue
max_valid_token = max(valid_tokens)
min_valid_token = min(valid_tokens)
print(f"Valid REMI token range: {min_valid_token} to {max_valid_token}")



vocab_size = max_valid_token + 3  # Add 3 for PAD, BOS, EOS tokens
block_size = 512
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MusicTransformer(vocab_size).to(device)

# Initialize dataset and check contents
print("\nInitializing dataset...")
dataset = MIDIDataset(new_dataset_path, GENRE_TO_ID, max_seq_len=512)


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(epochs=10):    
    # Start training
    print("\nStarting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)  # Ignore padding tokens in loss calculation

    total_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        # Create progress bar for this epoch
        progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', 
                          leave=True, unit='batch')
        
        for genre_input, dec_input, dec_target in progress_bar:
            # Move to device
            genre_input = genre_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            
            # Forward pass
            logits = model(genre_input, dec_input)
            
            # Compute loss
            loss = loss_fn(logits.view(-1, vocab_size), dec_target.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
    
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")


# Model saving/loading
model_save_path = "transformer.pth"

if not os.path.exists(model_save_path):
    print("Training model...")
    train(epochs=20)
    model.save_model(model_save_path)
else:
    print("Model already exists, loading pretrained model...")
    model = MusicTransformer.load_model(model_save_path, device=device)




print("Generating MIDI from scratch...")
model.eval()

num_candidates = 3
goal_length = 500

genre_to_generate = 'Jazz'
genre_token = f"<|{genre_to_generate}|>"

print(f"Generating {num_candidates} candidates...")

# Create output directory if it doesn't exist
dir = 'transformer/'
os.makedirs(dir, exist_ok=True)

all_outputs = []
for i in range(num_candidates):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Generate sequence using the model's generate method
        output = model.generate(genre_token, max_length=goal_length, temperature=1.0)
        all_outputs.append(output)
        print(f"Successfully generated candidate {i + 1}")
    except Exception as e:
        print(f"Error generating candidate {i + 1}: {str(e)}")

if all_outputs:
    output_files = []
    for idx, output in enumerate(all_outputs):
        print(f"Processing candidate {idx + 1}")
        
        try:
            # Remove special tokens (BOS, EOS, PAD) and prepare sequence
            # Get all tokens after BOS and before EOS/PAD
            tokens = output[0].cpu().numpy()
            tokens = tokens[1:]  # Remove BOS token
            
            # Find the first occurrence of EOS or PAD
            eos_idx = np.where((tokens == EOS) | (tokens == PAD))[0]
            if len(eos_idx) > 0:
                tokens = tokens[:eos_idx[0]]
            
            # Convert to 2D array as expected by tokenizer
            tokens = np.expand_dims(tokens, axis=0)
            print(f"Token shape before decoding: {tokens.shape}")
            
            # Decode to MIDI
            midi_obj = remi_tokenizer.decode(tokens)
            
            output_file = f"{dir}{genre_to_generate}_output_{idx + 1}.mid"
            midi_obj.dump_midi(output_file)
            
            output_files.append(output_file)
            print(f"Output MIDI for candidate {idx + 1} saved to {output_file}")
        except Exception as e:
            print(f"Error saving MIDI for candidate {idx + 1}: {str(e)}")
            print(f"Token sequence: {tokens}")
    
    if output_files:
        print("Playing all generated outputs sequentially...")
        for file in output_files:
            print(f"Now playing: {file}")
            try:
                musicPlayer = MusicPlayer(file)
                musicPlayer.play_music()
            except Exception as e:
                print(f"Error playing {file}: {str(e)}")
    else:
        print("No valid outputs were generated")
        sys.exit(1)
else:
    print("Failed to generate any valid outputs")
    sys.exit(1)
        