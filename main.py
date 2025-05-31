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

# Add this at the top of your file with other imports
sys.setrecursionlimit(100000)  # Increase recursion limit

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("Path to dataset files:", path)


basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def convert_wav_to_midi(wav_path, output_dir):
    # Run prediction
    try:
        model_output, midi_data, note_events = predict(
            wav_path,
            basic_pitch_model,
        )

        # Save MIDI
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        midi_path = os.path.join(output_dir, f"{base_name}.mid")
        midi_data.write(midi_path)
        print(f"[✓] Saved: {midi_path}")
    except Exception as e:
        print(f"Error: {e}")

def batch_convert(folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(wav_files)
    if not wav_files:
        print("No .wav files found.")
        return

    for wav_file in wav_files:
        convert_wav_to_midi(wav_file, output_dir)
# Example usage
input_folder = "Data/genres_original"
output_folder = "Data/midi_output"

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
if not os.path.exists(output_folder):
    for genre in genres:
        temp_input_folder = input_folder + "/" + genre
        temp_output_folder = output_folder + "/" + genre
        print(temp_input_folder)
        print(temp_output_folder)
        batch_convert(temp_input_folder, temp_output_folder)
else:
    print("Output folder already exists.")

# musicPlayer = MusicPlayer("gtzan-dataset-music-genre-classification/versions/1/Data/midi_output/blues/blues.00000.mid")
# musicPlayer.play_music()



config = TokenizerConfig()
remi_tokenizer = REMI(config)
dataset_path = "dataset.txt"

a = 0
def tokenize_midi_folder(midi_folder, output_file):
    genre = midi_folder.split("/")[-1]
    print(f'Tokenizing {genre}...')
    all_tokens = []

    for midi_path in os.listdir(midi_folder):
        if not midi_path.endswith(".mid"):
            continue
        midi = MidiFile(os.path.join(midi_folder, midi_path))
        # Add genre token at the start
        genre_token = f"<|{genre}|>"
        tokens = remi_tokenizer(midi) # List[TokSequence]
        # print(f'tokens: {tokens}')
        # print(f'tokens[0].ids: {tokens[0].ids}')
        # Convert REMI tokens to their integer values
        all_sequences = [token for token in tokens[0].ids]
        # Convert tokens to strings and add special tokens
        token_str = f"{genre_token} {' '.join(map(str, all_sequences))}"  # Add separator after first sequence
        all_tokens.append(token_str)

    # Save token sequences as dataset
    with open(output_file, "a", encoding='utf-8') as f:
        for sequence in all_tokens:
            f.write(sequence + "\n")

if not os.path.exists(dataset_path):
    for genre in genres:
        tokenize_midi_folder(output_folder + "/" + genre, dataset_path)
else:
    print(f"Dataset already exists in {dataset_path}")


class IdentityTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.genre_tokens = {f"<|{genre}|>": i + 2 for i, genre in enumerate([
            'blues', 'classical', 'country', 'disco', 'hiphop', 
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ])}
        # Increase vocab size to accommodate REMI tokens which can be quite large
        self.vocab_size = 30000  # Adjust this based on your actual token range
    
    def __call__(self, text, **kwargs):
        parts = text.strip().split()
        if not parts:
            return {
                'input_ids': torch.tensor([[self.pad_token_id]]),
                'attention_mask': torch.tensor([[1]])
            }

        # Handle genre token
        genre_token = parts[0]
        tokens = [self.genre_tokens.get(genre_token, 0)]
        
        # Convert remaining tokens to integers
        try:
            tokens.extend(int(t) for t in parts[1:])
        except ValueError as e:
            print(f"Error converting tokens: {e}")
            print(f"Problematic text: {text[:100]}...")  # Print first 100 chars
            raise

        # Handle padding
        max_length = kwargs.get('max_length', len(tokens))
        if kwargs.get('padding') == 'max_length':
            tokens = tokens[:max_length]  # Truncate if needed
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))  # Pad if needed

        input_ids = torch.tensor([tokens])
        attention_mask = torch.ones(input_ids.shape)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def encode(self, text, return_tensors=None):
        output = self(text)
        return output['input_ids']

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return ' '.join(map(str, token_ids))

    def save_pretrained(self, path):
        # No need to save anything for this tokenizer
        pass

    @classmethod
    def from_pretrained(cls, path):
        return cls()

# Set up tokenizer with padding
id_tokenizer = IdentityTokenizer()
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Create custom dataset class
class MusicDataset(Dataset):
    def __init__(self, id_tokenizer, file_path, block_size):
        self.examples = []
        self.block_size = block_size
        self.average_length = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        for text in texts:
            # Split genre token and sequence
            parts = text.strip().split()
            self.average_length += len(parts)
            if len(parts) < 2:  # Skip empty or invalid entries
                continue
                
            genre = parts[0]  # Get genre token
            sequence = ' '.join(parts[1:])  # Get actual sequence
            
            # Skip if sequence is too short
            if len(sequence.split()) < 10:  # Adjust minimum length as needed
                continue
            
            encodings = id_tokenizer(sequence, 
                                truncation=True,
                                max_length=block_size,
                                padding='max_length',
                                return_tensors='pt')
            
            # Skip if encoding failed or produced empty tensors
            if encodings['input_ids'].numel() == 0:
                continue
                
            example = {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze(),
                'genre': genre
            }
            
            # Verify tensor shapes are correct
            if (example['input_ids'].size(0) != block_size or 
                example['attention_mask'].size(0) != block_size or 
                example['labels'].size(0) != block_size):
                continue
                
            self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} valid examples out of {len(texts)} total entries")
        self.average_length /= len(self.examples)
        print(f"Average length of examples: {self.average_length}")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]





# Replace TextDataset with our custom dataset
dataset = MusicDataset(
    id_tokenizer=id_tokenizer,
    file_path=dataset_path,
    block_size=1024,
)

# Check for GPU availability
if torch.cuda.is_available():
    print("GPU available! Using:", torch.cuda.get_device_name(0))
    device = 'cuda'
    use_fp16 = True
else:
    print("No GPU available. Using CPU.")
    device = 'cpu'
    use_fp16 = False

training_args = TrainingArguments(
    output_dir="./music_transformer",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    no_cuda=(device == 'cpu'),  # Disable CUDA if no GPU
    fp16=use_fp16,             # Only use fp16 if GPU available
)

# Only move model to GPU if available
if device == 'cuda':
    model = model.cuda()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)



model_path = "music_transformer"

try:
    # Try loading model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    id_tokenizer = IdentityTokenizer.from_pretrained(model_path)
    print(f"✅ Model loaded from {model_path}")
except (OSError, FileNotFoundError) as e:
    print(f"⚠️ Could not load model from {model_path}. Reason: {e}")
    print("🧠 Starting training...")

    # Train and save
    trainer.train()
    model.save_pretrained(model_path)
    id_tokenizer.save_pretrained(model_path)
    print(f"💾 Model saved to {model_path}")







print("Evaluating...")
model.eval()






genre_to_generate = 'country'
genre_token = f"<|{genre_to_generate}|>"  # Correct format for genre token

# Create a more structured initial prompt
# Add some common musical tokens that appear in your training data
initial_tokens = [
    id_tokenizer.genre_tokens[genre_token],  # Genre token
    # 189,  # Common starting token from your training data
    # 18, 109, 129  # Add some common musical structure tokens
]
model_input = torch.tensor([initial_tokens]).to(device)
print(f'Initial prompt tokens: {initial_tokens}')

# Before generation, get the valid token range for REMI
print("Analyzing REMI tokenizer vocabulary...")
valid_tokens = set()
for token_id in range(2000):  # Check first 1000 potential tokens
    try:
        remi_tokenizer[token_id]
        valid_tokens.add(token_id)
    except:
        continue
max_valid_token = max(valid_tokens)
min_valid_token = min(valid_tokens)
print(f"Valid REMI token range: {min_valid_token} to {max_valid_token}")

# Generation parameters for more distinct outputs
generation_params = {
    'input_ids': model_input,
    'max_new_tokens': 10,  # Generate this many new tokens each iteration
    'temperature': 0.1,    # Lower temperature for more focused generation
    'top_k': 50,          # More focused sampling
    'top_p': 0.92,        # Slightly more focused nucleus sampling
    'do_sample': True,
    'num_return_sequences': 1,
    'pad_token_id': id_tokenizer.eos_token_id,
    'repetition_penalty': 1.2,  # Penalize token repetition
}

# Generate multiple sequences and pick the most diverse one
num_input_tokens = 10
num_candidates = 3
goal_length = 500
all_outputs = []
all_scores = []

print(f"Generating {num_candidates} candidates...")
for i in range(num_candidates):
    # Clear CUDA cache before each generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    valid_output = []
    current_input = model_input
    
    # Generate sequence in chunks
    while len(valid_output) < goal_length:
        try:
            sample_output = model.generate(**{**generation_params, 'input_ids': current_input})
            
            # Get only the new tokens (exclude the input tokens)
            new_tokens = sample_output[0][current_input.size(1):]
            
            # Filter and add valid tokens
            for token in new_tokens.cpu().numpy():
                if token in valid_tokens or token in id_tokenizer.genre_tokens.values():
                    valid_output.append(token)
            
            # Update input for next iteration with the last 5 valid tokens
            if len(valid_output) >= num_input_tokens:
                current_input = torch.tensor([valid_output[-num_input_tokens:]]).to(device)
            else:
                current_input = torch.tensor([valid_output]).to(device)
                
            print(f"Candidate {i+1} progress: {len(valid_output)}/{goal_length} tokens", end='\r')
            
        except Exception as e:
            print(f"\nError during generation: {e}")
            break
    
    print()  # New line after progress
    
    if len(valid_output) < 10:  # Skip if too few valid tokens
        print(f"Candidate {i+1} had too few valid tokens, retrying...")
        continue
        
    all_outputs.append(torch.tensor(valid_output))
    
    # Calculate diversity score (based on token variety)
    unique_tokens = len(set(valid_output))
    all_scores.append(unique_tokens)
    print(f"Candidate {i+1} valid tokens: {len(valid_output)}, unique tokens: {unique_tokens}")

if not all_outputs:
    print("Failed to generate any valid sequences. Trying with more conservative parameters...")
    generation_params.update({
        'temperature': 1.0,
        'top_k': 30,
        'max_new_tokens': 256,
    })
    sample_output = model.generate(**{**generation_params, 'input_ids': model_input})
    valid_output = [t for t in sample_output[0].cpu().numpy() if t in valid_tokens or t in id_tokenizer.genre_tokens.values()]
    if len(valid_output) >= 10:
        all_outputs.append(torch.tensor(valid_output))
        all_scores.append(len(set(valid_output)))


if all_outputs:
    output_files = []
    for idx, output in enumerate(all_outputs):
        sample_output = output.unsqueeze(0)
        print(f"Processing candidate {idx + 1} with {all_scores[idx]} unique tokens")
        
        # Validate token list before decoding
        token_list = sample_output[0].tolist()
        print(f"Token range in output: {min(token_list)} to {max(token_list)}")
        
        # Additional validation before REMI decoding
        token_list = [t for t in token_list if t in valid_tokens or t in id_tokenizer.genre_tokens.values()]
        if len(token_list) < 10:
            print(f"Warning: Not enough valid tokens for candidate {idx + 1}, skipping")
            continue
        
        # Generate MIDI file for this output
        decoded = id_tokenizer.decode(sample_output[0])
        decoded_clean = decoded.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()
        token_list = [int(token) for token in decoded_clean.split() if token.isdigit()]
        score = remi_tokenizer.decode([token_list])
        print(f"Score {idx + 1}: {score}")
        
        output_file = f"{genre_to_generate}_output_{idx + 1}.mid"
        try:
            if hasattr(score, 'dump_midi'):
                score.dump_midi(output_file)
            else:
                print(f"Error: Score object for candidate {idx + 1} doesn't have a standard save method")
                continue
            output_files.append(output_file)
            print(f"Output MIDI for candidate {idx + 1} saved to {output_file}")
        except Exception as e:
            print(f"Error saving MIDI for candidate {idx + 1}: {str(e)}")
    
    if output_files:
        print("Playing all generated outputs sequentially...")
        for file in output_files:
            print(f"Now playing: {file}")
            musicPlayer = MusicPlayer(file)
            musicPlayer.play_music()
    else:
        print("No valid outputs were generated")
        sys.exit(1)
else:
    print("Failed to generate any valid outputs")
    sys.exit(1)

# decoded = id_tokenizer.decode(sample_output[0])

# print(sample_output)
# print(decoded)

# # Step 1: Remove special tokens
# decoded_clean = decoded.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()

# # Step 2: Convert text back into tokens
# token_list = []
# for token in decoded_clean.split():
#     try:
#         token_list.append(int(token))
#     except ValueError:
#         print(f"⚠️ Skipping non-integer token: {token}")

# print(f'token_list: {token_list}')

# # Step 3: Use REMI tokenizer to decode tokens into MIDI
# score = remi_tokenizer.decode([token_list])

# print(f'Type of score: {type(score)}')
# print(f'Score contents: {score}')

# # Step 4: Save the MIDI directly
# outputMidiPath = "output.mid"
# print("Score object methods:", dir(score))

# try:
#     # Try different methods to save depending on the score type
#     if hasattr(score, 'write'):
#         score.write(outputMidiPath)
#     elif hasattr(score, 'dump'):
#         score.dump(outputMidiPath)
#     elif hasattr(score, 'save'):
#         score.save(outputMidiPath)
#     elif hasattr(score, 'dump_midi'):
#         score.dump_midi(outputMidiPath)
#     else:
#         print("Error: Score object doesn't have a standard save method")
#         print("Available methods:", dir(score))
# except Exception as e:
#     print(f"Error saving MIDI: {str(e)}")
#     print("Score object methods:", dir(score))

# print(f"Output MIDI saved to {outputMidiPath}, playing...")
# musicPlayer = MusicPlayer(outputMidiPath)
# musicPlayer.play_music()


