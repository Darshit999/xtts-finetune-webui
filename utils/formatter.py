import gc
import os

import pandas
import torch
import torchaudio
from faster_whisper import WhisperModel
from tqdm import tqdm

# torch.set_num_threads(1)
from utils.tokenizer import multilingual_cleaners

torch.set_num_threads(16)

from glob import glob

audio_types = (".wav", ".mp3", ".flac")

def find_latest_best_model(folder_path):
    search_path = os.path.join(folder_path, '**', 'best_model.pth')
    files = glob(search_path, recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

def list_audios(basePath, contains=None):
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):
                audioPath = os.path.join(rootDir, filename)
                yield audioPath

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

    lang_file_path = os.path.join(out_path, "lang.txt")

    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print(f"Warning, existing language({current_language}) does not match target language({target_language}). Updated lang.txt with target language.")
    else:
        print(f"Existing language({current_language}) matches target language({target_language}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Whisper Model!")
    asr_model = WhisperModel(whisper_model, device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    tqdm_object = tqdm(audio_files) if gradio_progress is None else gradio_progress.tqdm(audio_files, desc="Formatting...")

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)

        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        segments = list(segments)

        i = 0
        sentence = ""
        sentence_start = None
        first_word = True

        words_list = []
        for _, segment in enumerate(segments):
            words_list.extend(list(segment.words))

        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start

                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                next_word_start = words_list[word_idx + 1].start if word_idx + 1 < len(words_list) else (wav.shape[0] - 1) / sr

                word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)

                if audio.size(-1) >= sr / 3:
                    torchaudio.save(absoulte_path, audio, sr)
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

    df = pandas.DataFrame(metadata).sample(frac=1)
    num_val_samples = int(len(df) * eval_percentage)

    df_eval, df_train = df[:num_val_samples], df[num_val_samples:]

    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.sort_values('audio_file').to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval.sort_values('audio_file').to_csv(eval_metadata_path, sep="|", index=False)

    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size
