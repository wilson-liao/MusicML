import kagglehub
import os
import glob
import pretty_midi
import librosa
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from musicPlayer import MusicPlayer
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset
from torch.utils.data import Dataset
import torch

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("Path to dataset files:", path)


basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

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
input_folder = "C:/Users/wilso/OneDrive/Desktop/MusicML/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original"
output_folder = "C:/Users/wilso/OneDrive/Desktop/MusicML/gtzan-dataset-music-genre-classification/versions/1/Data/midi_output"

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


# musicPlayer = MusicPlayer("C:/Users/wilso/OneDrive/Desktop/MusicML/gtzan-dataset-music-genre-classification/versions/1/Data/midi_output/blues/blues.00000.mid")
# musicPlayer.play_music()



config = TokenizerConfig()
tokenizer = REMI(config)
dataset_path = "dataset.txt"

def tokenize_midi_folder(midi_folder, output_file):
    genre = midi_folder.split("/")[-1]
    print(f'Tokenizing {genre}...')
    all_tokens = []

    for midi_path in os.listdir(midi_folder):
        if not midi_path.endswith(".mid"):
            continue
        midi = MidiFile(os.path.join(midi_folder, midi_path))
        tokens = tokenizer(midi)
        # Convert tokens to string and add special tokens
        token_str = f"<|startoftext|>{' '.join(map(str, tokens))}<|endoftext|>"
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


# Set up tokenizer with padding
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match tokenizer

# Create custom dataset class
class MusicDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        for text in texts:
            encodings = tokenizer(text.strip(), 
                                truncation=True,
                                max_length=block_size,
                                padding='max_length',
                                return_tensors='pt')
            self.examples.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            })

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

print("Training...")
# print(dataset[0])

trainer.train()

model_path = "music_transformer"

if not os.path.exists(model_path):
    model.save_pretrained(model_path)
else:
    print(f"Model already exists in {model_path}")


input_ids = tokenizer.encode("Genre_jazz", return_tensors="pt")
sample_output = model.generate(input_ids, max_length=512, temperature=1.0, top_k=50)
decoded = tokenizer.decode(sample_output[0])

outputMidiPath = "output.mid"
decoded.dump(outputMidiPath)

print(f"Output MIDI saved to {outputMidiPath}, playing...")
musicPlayer = MusicPlayer(outputMidiPath)
musicPlayer.play_music()
