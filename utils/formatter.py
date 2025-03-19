import gc
import os
import pandas as pd
import torch
import torchaudio
from faster_whisper import WhisperModel
from tqdm import tqdm

from utils.tokenizer import multilingual_cleaners
from glob import glob

torch.set_num_threads(16)

audio_types = (".wav", ".mp3", ".flac")

def find_latest_best_model(folder_path):
    search_path = os.path.join(folder_path, '**', 'best_model.pth')
    files = glob(search_path, recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

def list_audios(basePath, contains=None):
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    for (rootDir, _, filenames) in os.walk(basePath):
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
    out_path="/content/output",
    buffer=0.2,
    eval_percentage=0.15,
    speaker_name="coqui",
    gradio_progress=None
):
    audio_total_size = 0

    os.makedirs(out_path, exist_ok=True)

    # Ensure the wavs folder is created inside the output directory
    wavs_folder = os.path.join(out_path, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    lang_file_path = os.path.join(out_path, "lang.txt")

    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            current_language = f.read().strip()

    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as f:
            f.write(target_language + '\n')
        print(f"Updated lang.txt with target language: {target_language}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper Model...")
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    tqdm_object = tqdm(audio_files, desc="Processing Audio Files")

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)

        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        words_list = [word for segment in segments for word in segment.words]

        sentence, sentence_start, first_word = "", None, True
        for i, word in enumerate(words_list):
            if first_word:
                sentence_start = max(word.start - buffer, 0)
                if i > 0:
                    previous_end = words_list[i - 1].end
                    sentence_start = max(sentence_start, (previous_end + sentence_start) / 2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]  # Remove leading space
                sentence = multilingual_cleaners(sentence, target_language)

                audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]
                audio_file = f"{audio_file_name}_{str(i).zfill(8)}.wav"
                abs_audio_path = os.path.join(wavs_folder, audio_file)

                next_start = words_list[i + 1].start if i + 1 < len(words_list) else (wav.shape[0] - 1) / sr
                word_end = min((word.end + next_start) / 2, word.end + buffer)

                audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)

                if audio.size(-1) >= sr / 3:
                    torchaudio.save(abs_audio_path, audio, sr)
                    metadata["audio_file"].append(f"wavs/{audio_file}")
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)

                sentence, first_word = "", True

    df = pd.DataFrame(metadata).sample(frac=1)
    num_val_samples = int(len(df) * eval_percentage)

    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    df.iloc[num_val_samples:].to_csv(train_metadata_path, sep="|", index=False)
    df.iloc[:num_val_samples].to_csv(eval_metadata_path, sep="|", index=False)

    del asr_model, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size
