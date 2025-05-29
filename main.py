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
tokenizer = REMI(config)
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
        tokens = tokenizer(midi) # List[TokSequence]
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
tokenizer = IdentityTokenizer()
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Create custom dataset class
class MusicDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        for text in texts:
            # Split genre token and sequence
            parts = text.strip().split()
            genre = parts[0]  # Get genre token
            sequence = ' '.join(parts[1:])  # Get actual sequence
            
            encodings = tokenizer(sequence, 
                                truncation=True,
                                max_length=block_size,
                                padding='max_length',
                                return_tensors='pt')
            
            example = {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze(),
                'genre': genre
            }
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]





# Replace TextDataset with our custom dataset
dataset = MusicDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=512,
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
    per_device_train_batch_size=2,
    num_train_epochs=5,
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
    tokenizer = IdentityTokenizer.from_pretrained(model_path)
    print(f"✅ Model loaded from {model_path}")
except (OSError, FileNotFoundError) as e:
    print(f"⚠️ Could not load model from {model_path}. Reason: {e}")
    print("🧠 Starting training...")

    # Train and save
    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"💾 Model saved to {model_path}")

print("Evaluating...")
model.eval()


input_ids = tokenizer.encode("Genre_jazz", return_tensors="pt")
inputs = input_ids.to(device)
sample_output = model.generate(inputs, max_length=512, temperature=1.0, top_k=50, pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.decode(sample_output[0])

# Step 1: Remove special tokens
decoded_clean = decoded.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()

# Step 2: Convert text back into tokens
token_list = []
for token in decoded_clean.split():
    try:
        token_list.append(int(token))
    except ValueError:
        print(f"⚠️ Skipping non-integer token: {token}")

# Step 3: Use REMI tokenizer to decode tokens into MIDI
remi_tokenizer = REMI(TokenizerConfig())
midi_obj: MidiFile = remi_tokenizer.decode(token_list)

# Step 4: Save the MIDI
outputMidiPath = "output.mid"
midi_obj.dump(outputMidiPath)


print(f"Output MIDI saved to {outputMidiPath}, playing...")
musicPlayer = MusicPlayer(outputMidiPath)
musicPlayer.play_music()
