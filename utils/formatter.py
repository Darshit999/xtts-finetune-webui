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
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath

def split_long_sentence(words_list, word_idx_start, word_idx_end, max_duration=12.0, min_split_duration=6.0):
    """
    Split long sentences based on duration and commas.
    
    Args:
        words_list: List of word objects with start and end times
        word_idx_start: Starting word index
        word_idx_end: Ending word index (inclusive)
        max_duration: Maximum allowed duration in seconds
        min_split_duration: Minimum duration for split chunks
        
    Returns:
        List of tuples with (start_idx, end_idx) for each split segment
    """
    # Calculate total duration
    if word_idx_end >= len(words_list):
        word_idx_end = len(words_list) - 1
        
    start_time = words_list[word_idx_start].start
    end_time = words_list[word_idx_end].end
    duration = end_time - start_time
    
    # If duration is acceptable, return the original segment
    if duration <= max_duration:
        return [(word_idx_start, word_idx_end)]
    
    splits = []
    current_start_idx = word_idx_start
    
    while current_start_idx <= word_idx_end:
        # Find a suitable end point
        suitable_end_idx = current_start_idx
        comma_candidates = []
        
        # Scan forward to find potential split points (commas)
        for i in range(current_start_idx, word_idx_end + 1):
            current_duration = words_list[i].end - words_list[current_start_idx].start
            
            # If we've exceeded max duration, stop looking
            if current_duration > max_duration:
                break
                
            # Record any commas we find that are at least min_split_duration away from start
            if "," in words_list[i].word and (words_list[i].end - words_list[current_start_idx].start) >= min_split_duration:
                comma_candidates.append(i)
            
            suitable_end_idx = i
            
            # If we've reached a good minimum duration, we have a valid fallback end point
            if current_duration >= min_split_duration:
                # If we're close to max_duration, stop to avoid overshooting
                if current_duration >= 0.8 * max_duration:
                    break
        
        # Choose split point - prefer comma if available within reasonable duration
        split_end_idx = suitable_end_idx
        if comma_candidates:
            # Take the last comma candidate (closest to max_duration while still valid)
            split_end_idx = comma_candidates[-1]
        
        splits.append((current_start_idx, split_end_idx))
        current_start_idx = split_end_idx + 1
        
    return splits

def format_audio_list(
    audio_files, 
    target_language="en", 
    whisper_model = "large-v3", 
    out_path=None, 
    buffer=0.2, 
    eval_percentage=0.15, 
    speaker_name="coqui", 
    gradio_progress=None,
    compute_type="float32"
    ):
    audio_total_size = 0
    # make sure that ooutput file exists
    os.makedirs(out_path, exist_ok=True)

    # Write the target language to lang.txt in the output directory
    lang_file_path = os.path.join(out_path, "lang.txt")
    
    # Check if lang.txt already exists and contains a different language
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
    
    if current_language != target_language:
        # Only update lang.txt if target language is different from current language
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print(f"Warning, existing language({current_language}) does not match target language({target_language}). Updated lang.txt with target language.")
    else:
        print(f"Existing language({current_language}) matches target language({target_language}).")

    # Loading Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    print("Loading Whisper Model!")
    asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        segments = list(segments)
        # print(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # Process for building sentences
        sentence_words_indices = []  # To store start and end indices of words in a sentence
        word_idx = 0
        
        while word_idx < len(words_list):
            # Start a new sentence
            sentence_start_idx = word_idx
            
            # Scan until end of sentence marker
            while word_idx < len(words_list) and not (word_idx > 0 and words_list[word_idx-1].word[-1] in ["!","。", ".", "?", "।"]):
                word_idx += 1
            
            # Include the last word if we've found an end marker or reached the end
            if word_idx < len(words_list):
                word_idx += 1  # Include the word with the end marker
                
            sentence_end_idx = min(word_idx - 1, len(words_list) - 1)
            
            # Check if this is a valid sentence
            if sentence_end_idx >= sentence_start_idx:
                # Get start and end times to check duration
                start_time = words_list[sentence_start_idx].start
                end_time = words_list[sentence_end_idx].end
                
                # If sentence is too long, split it
                if end_time - start_time > 12.0:  # 12 seconds max duration
                    # Use helper function to split the sentence
                    splits = split_long_sentence(words_list, sentence_start_idx, sentence_end_idx)
                    sentence_words_indices.extend(splits)
                else:
                    # Normal-length sentence
                    sentence_words_indices.append((sentence_start_idx, sentence_end_idx))
            
        # Process each identified sentence or split segment
        for sentence_start_idx, sentence_end_idx in sentence_words_indices:
            # Build the sentence text
            sentence = "".join([words_list[j].word for j in range(sentence_start_idx, sentence_end_idx + 1)])
            
            # Get the sentence timing
            sentence_start = words_list[sentence_start_idx].start
            
            # If it is the first sentence in the audio, add buffer
            if sentence_start_idx == 0:
                sentence_start = max(sentence_start - buffer, 0)  # Add buffer to the sentence start
            else:
                # Get previous sentence end
                previous_word_end = words_list[sentence_start_idx - 1].end
                # Add buffer or get the silence middle between the previous sentence and the current one
                sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start)/2)

            # Clean up the sentence
            sentence = sentence[1:] if sentence and sentence[0] == " " else sentence
            # Expand number and abbreviations plus normalization
            sentence = multilingual_cleaners(sentence, target_language)
            
            # Skip empty sentences
            if not sentence:
                continue
                
            audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
            audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

            # Check for the next word's existence to determine end time
            if sentence_end_idx + 1 < len(words_list):
                next_word_start = words_list[sentence_end_idx + 1].start
            else:
                # If no more words, use the audio length as next word start
                next_word_start = (wav.shape[0] - 1) / sr

            # Average the current word end and next word start
            word_end = min((words_list[sentence_end_idx].end + next_word_start) / 2, words_list[sentence_end_idx].end + buffer)
            
            absoulte_path = os.path.join(out_path, audio_file)
            os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
            i += 1

            audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
            # Calculate audio duration in seconds
            audio_duration = audio.size(-1) / sr
            
            torchaudio.save(absoulte_path,
                audio,
                sr
            )
            
            metadata["audio_file"].append(audio_file)
            metadata["text"].append(sentence)
            metadata["speaker_name"].append(speaker_name)

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del asr_model, df_train, df_eval, df, metadata
    gc.collect()
    
    return train_metadata_path, eval_metadata_path, audio_total_size
