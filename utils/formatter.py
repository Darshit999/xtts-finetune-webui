import gc
import os
import pandas as pd
import torch
import torchaudio
from faster_whisper import WhisperModel
from tqdm import tqdm
from glob import glob
from utils.tokenizer import multilingual_cleaners

# Set the number of threads for Torch
torch.set_num_threads(16)

# Define valid audio types
audio_types = (".wav", ".mp3", ".flac")

# Ensure output paths
BASE_PATH = "/content/output"
WAVS_PATH = os.path.join(BASE_PATH, "wavs")
os.makedirs(WAVS_PATH, exist_ok=True)

def find_latest_best_model(folder_path):
    search_path = os.path.join(folder_path, '**', 'best_model.pth')
    files = glob(search_path, recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

def list_files(basePath, validExts=None, contains=None):
    """ Lists all valid audio files in a directory. """
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains and contains not in filename:
                continue
            ext = filename[filename.rfind("."):].lower()
            if validExts is None or ext.endswith(validExts):
                yield os.path.join(rootDir, filename)

def format_audio_list(
    audio_files, 
    target_language="en", 
    whisper_model="large-v3", 
    out_path=BASE_PATH, 
    buffer=0.2, 
    eval_percentage=0.15, 
    speaker_name="coqui", 
    gradio_progress=None
):
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    # Create lang.txt
    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
    
    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print(f"Warning, existing language({current_language}) does not match target language({target_language}). Updated lang.txt.")

    # Load Whisper Model
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print("Loading Whisper Model...")
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...") if gradio_progress else tqdm(audio_files)

    for audio_path in tqdm_object:
        try:
            # Load audio file
            wav, sr = torchaudio.load(audio_path)
            print(f"Loaded: {audio_path} | Length: {wav.size(-1) / sr:.2f}s | Sample Rate: {sr}")

            if wav.size(0) != 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            wav = wav.squeeze()
            audio_total_size += (wav.size(-1) / sr)

            # Transcribe
            segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
            segments = list(segments)
            print(f"Transcription for {audio_path}: {segments}")

            if not segments:
                print(f"Skipping {audio_path} - No transcription found!")
                continue  # Skip files with no transcription

            # Process words
            i = 0
            words_list = []
            for segment in segments:
                words_list.extend(segment.words)

            first_word = True
            sentence = ""
            sentence_start = None

            for word_idx, word in enumerate(words_list):
                if first_word:
                    sentence_start = max(word.start - buffer, 0) if word_idx == 0 else (words_list[word_idx - 1].end + word.start) / 2
                    sentence = word.word
                    first_word = False
                else:
                    sentence += word.word

                if word.word[-1] in ["!", ".", "?"]:
                    sentence = sentence[1:]
                    sentence = multilingual_cleaners(sentence, target_language)
                    audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]
                    audio_file = os.path.join(WAVS_PATH, f"{audio_file_name}_{str(i).zfill(8)}.wav")

                    next_word_start = words_list[word_idx + 1].start if word_idx + 1 < len(words_list) else (wav.shape[0] - 1) / sr
                    word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
                    i += 1
                    first_word = True

                    audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)

                    if audio.size(-1) >= sr / 3:
                        torchaudio.save(audio_file, audio, sr)
                        print(f"Saved: {audio_file} | Length: {audio.size(-1) / sr:.2f}s")
                    else:
                        continue

                    metadata["audio_file"].append(audio_file)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Convert metadata to CSV
    df = pd.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df) * eval_percentage)

    df_eval, df_train = df[:num_val_samples], df[num_val_samples:]

    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    df_train.to_csv(train_metadata_path, sep="|", index=False)
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    print(f"Train CSV: {train_metadata_path} ({len(df_train)} entries)")
    print(f"Eval CSV: {eval_metadata_path} ({len(df_eval)} entries)")

    # Clean up memory
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size
