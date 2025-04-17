import gc
import os
import time
import shutil
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

def create_temporary_file(folder, suffix=".wav"):
    """Create a temporary file path in the specified folder."""
    import uuid
    temp_filename = f"temp_{uuid.uuid4()}{suffix}"
    return os.path.join(folder, temp_filename)

def format_audio_list(
    audio_files, 
    target_language="en", 
    whisper_model="large-v3", 
    out_path=None, 
    buffer=0.2, 
    eval_percentage=0.15, 
    speaker_name="coqui", 
    gradio_progress=None,
    compute_type="float32"
    ):
    audio_total_size = 0
    # make sure that output file exists
    os.makedirs(out_path, exist_ok=True)
    
    # Create temp folder for processing
    temp_folder = os.path.join(out_path, "temp")
    os.makedirs(temp_folder, exist_ok=True)

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
    
    # Check for existing metadata
    existing_metadata = {'train': None, 'eval': None}
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pandas.read_csv(train_metadata_path, sep="|")
        print("Existing training metadata found and loaded.")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pandas.read_csv(eval_metadata_path, sep="|")
        print("Existing evaluation metadata found and loaded.")

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        # Handle string path or file-like object
        if isinstance(audio_path, str):
            audio_file_name_without_ext, _ = os.path.splitext(os.path.basename(audio_path))
            audio_path_name = audio_path
        elif hasattr(audio_path, 'read'):
            # If it has a 'read' attribute, treat it as a file-like object
            audio_file_name_without_ext, _ = os.path.splitext(os.path.basename(audio_path.name))
            audio_path_name = create_temporary_file(temp_folder)
            with open(audio_path, 'rb') as original_file:
                file_content = original_file.read()
            with open(audio_path_name, 'wb') as temp_file:
                temp_file.write(file_content)
        
        # Create a temporary file path within the temp folder
        temp_audio_path = create_temporary_file(temp_folder)
        
        try:
            if isinstance(audio_path, str):
                audio_path_name = audio_path
            elif hasattr(audio_path, 'name'):
                audio_path_name = audio_path.name
            else:
                raise ValueError(f"Unsupported audio_path type: {type(audio_path)}")
        except Exception as e:
            print(f"Error reading original file: {e}")
            continue
            
        print(f"Current working file: {audio_path_name}")
        
        try:
            # Copy the audio content
            time.sleep(0.5)  # Introduce a small delay
            shutil.copy2(audio_path_name, temp_audio_path)
        except Exception as e:
            print(f"Error copying file: {e}")
            continue
            
        # Check if this file has already been processed
        prefix_check = f"wavs/{audio_file_name_without_ext}_"
        
        # Check both training and evaluation metadata for an entry that starts with the file name
        skip_processing = False
        
        for key in ['train', 'eval']:
            if existing_metadata[key] is not None:
                mask = existing_metadata[key]['audio_file'].str.startswith(prefix_check)
                
                if mask.any():
                    print(f"Segments from {audio_file_name_without_ext} have been previously processed; skipping...")
                    skip_processing = True
                    break
                    
        # If we found that we've already processed this file before, continue to the next iteration
        if skip_processing:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            continue
            
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # process each word
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                # If it is the first sentence, add buffer or get the begining of the file
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)  # Add buffer to the sentence start
                else:
                    # get previous sentence end
                    previous_word_end = words_list[word_idx - 1].end
                    # add buffer or get the silence midle between the previous sentence and the current one
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start)/2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word
            
            if word.word[-1] in ["!","。", ".", "?", "।"]:
                sentence = sentence[1:]
                # Expand number and abbreviations plus normalization
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Check for the next word's existence
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    # If don't have more words it means that it is the last sentence then use the audio len as next word start
                    next_word_start = (wav.shape[0] - 1) / sr

                # Average the current word end and next word start
                word_end = min((word.end + next_word_start) / 2, word.end + buffer)
                
                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
                # Calculate audio duration in seconds
                audio_duration = audio.size(-1) / sr
                
                # Skip clips that are too short (less than 0.33 seconds)
                if audio_duration >= 1/3:
                    torchaudio.save(absoulte_path,
                        audio,
                        sr
                    )
                    
                    metadata["audio_file"].append(audio_file)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)
                
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # Handle existing metadata if it exists
    if os.path.exists(train_metadata_path) and os.path.exists(eval_metadata_path):
        existing_train_df = existing_metadata['train']
        existing_eval_df = existing_metadata['eval']
    else:
        existing_train_df = pandas.DataFrame(columns=["audio_file", "text", "speaker_name"])
        existing_eval_df = pandas.DataFrame(columns=["audio_file", "text", "speaker_name"])

    new_data_df = pandas.DataFrame(metadata)
    
    # Combine existing and new data
    combined_train_df = pandas.concat([existing_train_df, new_data_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # Shuffle and split the data
    combined_train_df_shuffled = combined_train_df.sample(frac=1)
    num_val_samples = int(len(combined_train_df_shuffled) * eval_percentage)
    
    final_eval_set = combined_train_df_shuffled[:num_val_samples]
    final_training_set = combined_train_df_shuffled[num_val_samples:]
    
    # Save the datasets
    final_training_set.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
    final_eval_set.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    # deallocate VRAM and RAM
    del asr_model, final_eval_set, final_training_set, new_data_df, existing_metadata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Train CSV: {train_metadata_path}")
    print(f"Eval CSV: {eval_metadata_path}")
    print(f"Audio Total Size: {audio_total_size}")
    
    return train_metadata_path, eval_metadata_path, audio_total_size
