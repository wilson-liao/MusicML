import kagglehub

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("Path to dataset files:", path)


from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch import audio_to_input
from basic_pitch.utils import save_midi
import os

def convert_wav_to_midi(wav_path, output_dir):
    # Load and process audio
    audio, sample_rate = audio_to_input(wav_path)
    
    # Predict pitch activations
    model_output, midi_data = predict(audio, sample_rate, model_path=ICASSP_2022_MODEL_PATH)
    
    # Create output path
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    midi_path = os.path.join(output_dir, f"{base_name}.mid")
    
    # Save MIDI
    save_midi(midi_data, midi_path)
    print(f"Saved: {midi_path}")



# Example usage
convert_wav_to_midi("path/to/input.wav", "path/to/output/dir")