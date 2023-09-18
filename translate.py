from googletrans import Translator
import nltk
import sys
import signal
import re
import json

from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS

import speech_recognition as sr
import numpy as np

import vosk


# global variable for locales
locale_dict = {}

#vosk_model = vosk.Model("C:\\Users\\ickylevel\\work\\dualtranslate\\model")


# Create a sine wave with increased amplitude for better audibility
frequency = 440  # Frequency of the beep sound (440 Hz is a common choice)
duration = 250  # Duration of the beep in milliseconds (1 second in this case)
sample_rate = 44100  # Samples per second

# Generate the sine wave
t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# Convert to integer audio data
audio_data = (sine_wave * 32767).astype(np.int16)

beep_sound = AudioSegment(
    audio_data.tobytes(),
    frame_rate=sample_rate,
    sample_width=audio_data.dtype.itemsize,
    channels=1,
)

frequency = 240
duration = 150
# Generate the sine wave
t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# Convert to integer audio data
audio_data = (sine_wave * 32767).astype(np.int16)

beep_sound2 = AudioSegment(
    audio_data.tobytes(),
    frame_rate=sample_rate,
    sample_width=audio_data.dtype.itemsize,
    channels=1,
)

audio_fail = AudioSegment.from_mp3("1.mp3")
audio_success = AudioSegment.from_mp3("2.mp3")
audio_end = AudioSegment.from_mp3("3.mp3")

recognizer = sr.Recognizer()

def read_info(text, language):
    if (not (text in locale_dict)):
        translator = Translator()
        try:
            locale_dict[text] = translator.translate(text, src='en', dest=language).text
        except:
            return
    print(locale_dict[text])
    read_text(locale_dict[text], language, False)
    
def test_vocabulary(word_list, trans_list, start_i, end_i, language, target_lang):
    for i in range(start_i, end_i):
        if (len(word_list[i]) > 2):
            print(f"Say the traduction for '{trans_list[i]}' in {language}:")
            read_info("Find the translation for: ", target_lang)
            read_text(trans_list[i], target_lang, False)
            compare_transcription(word_list[i], language, target_lang, 2)
            
def compare_transcription(original_word, source_language, target_lang, max_attempts):

    original_word = original_word.replace('.', '')
    
    recognizer.energy_threshold = 250 
    #recognizer.dynamic_energy_threshold = True  

    print(f"Say the word '{original_word}' in {source_language}:")
    read_info("Say the word after the beep", target_lang)
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")
        
        play(beep_sound)
        with sr.Microphone() as source:
            try:
                if (attempt == 0):
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=4)
                else:
                    audio = recognizer.record(source, duration=4)
            except:
                
                read_info("No Sound detected.", target_lang)
                play_recorded_audio(audio)
                
                play(beep_sound)
                try:
                    audio = recognizer.record(source, duration=5)
                except:
                    continue

        try:
            play(beep_sound2)
            # parsed_response = recognizer.recognize_vosk(audio, language=source_language)
            # recognized_text = json.loads(parsed_response)["text"]
            recognized_text = recognizer.recognize_google(audio, language=source_language)
            read_info("You said:", target_lang)
            print("You said:", recognized_text)
            read_text(recognized_text, source_language, False)
            if recognized_text.lower() == original_word.lower():
                print("Correct! You pronounced it correctly.")
                play(audio_success)
                return  # Exit the loop if the word is pronounced correctly
            else:
                print("Incorrect. Your pronunciation differs from the original word.")
                play(audio_fail)
                #read_info("Incorrect! Your pronunciation differs.", target_lang)

        except sr.UnknownValueError:
            read_info("Sorry, I couldn't understand your speech.", target_lang)
            play_recorded_audio(audio)
            print("Sorry, I couldn't understand your speech.")
        except sr.RequestError as e:
            read_info("Error.", target_lang)
            print(f"Could not request results from Google Speech Recognition service; {e}")
    read_info("The word was: ", target_lang)
    read_text(original_word, source_language, False)
    print(f"Maximum attempts reached ({max_attempts}). Moving on...")
    
    
def play_recorded_audio(recorded):
    temp_audio_file = "temp_audio.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(recorded.get_wav_data())
    audio = AudioSegment.from_wav(temp_audio_file)
    play(audio)

def read_text(text, language, do_slow):
    if (text != ""):
        tts = gTTS(text, lang=language, slow=do_slow)  
        print(f"Reading ({language}): {text}")
        tts.save("temp.mp3")
        audio = AudioSegment.from_mp3("temp.mp3")
        play(audio)

def translate_to_french(source_file, dest_file, decompose, read, source_lang, source_lang_full, target_lang, repeat_count):
    translator = Translator()
    nltk.download('punkt')  # Download the punkt tokenizer data if not already downloaded

    with open(source_file, 'r', encoding='utf-8') as file:
        original_text = file.read()

    # Use the NLTK tokenizer for sentence detection
    sentences = nltk.sent_tokenize(original_text, language=source_lang_full)

    processed_sentences = []
    translated_sentences = []
    total_sentences = len(sentences)

    print(" \n Translation in progress... (Press Ctrl+C to interrupt and save the translated content) \n")

    interrupted = False

    def handle_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nTranslation interrupted by user. Saving the translated content up to this point... Please Wait...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        for i, sentence in enumerate(sentences):
            if interrupted:
                break

            iter = len(translated_sentences)
            
            
            translated_words = []
            processed_words = []
            sentence = sentence.strip()
            if sentence:
            
                # per group translation:
                sub_sentences = re.split(r'[-–,]', sentence)
                # test to avoid reading twice the same
                if (len(sub_sentences) != 1):
                    try:
                        translation = translator.translate(sentence, src=source_lang, dest=target_lang)
                        translated_sentences.append(translation.text)
                        processed_sentences.append(sentence)
                        if read:
                            read_text(sentence, source_lang, True)
                            if interrupted:
                               break
                            read_text(translation.text, target_lang, False)
                            if interrupted:
                               break
                    except:
                        translated_sentences.append("[Translation failed]")
                
                for j, sub_sentence in enumerate(sub_sentences):
                    try:
                        translation = translator.translate(sub_sentence, src=source_lang, dest=target_lang)
                        translated_sentences.append(translation.text)
                        processed_sentences.append(sub_sentence)
                        if read:
                            read_text(sub_sentence, source_lang, True)
                            if interrupted:
                               break
                            read_text(translation.text, target_lang, False)
                            if interrupted:
                               break
                    except:
                        translated_sentences.append("[Translation failed]")
                    
                    # add a per word translation:
                    if decompose:
                        words = sub_sentence.split()
                        if (len(words) > 1):
                            for word in words:
                                if interrupted:
                                    break
                                try:
                                    translation = translator.translate(word, src=source_lang, dest=target_lang)
                                    
                                    processed_sentences.append(word)
                                    translated_sentences.append(translation.text)
                                    
                                    processed_words.append(word)
                                    translated_words.append(translation.text)
                                    
                                    if read:
                                        read_text(word, source_lang, True)
                                        read_text(translation.text, target_lang, False)
                                        #compare_transcription(word, source_lang, target_lang, 2)
                                except:
                                    processed_sentences.append(word)
                                    translated_sentences.append("[Translation failed]")
                
                
            for j in range(0, repeat_count):    
                for i in range(iter, min(len(translated_sentences), len(processed_sentences))):
                    read_text(processed_sentences[i], source_lang, True)
                    read_text(translated_sentences[i], target_lang, False)    

            test_vocabulary(processed_words, translated_words, 0, len(processed_words) - 1, source_lang, target_lang)
            
            play(audio_end)
            
            # Calculate completion percentage and print the status
            completion_percentage = (i + 1) / total_sentences * 100
            print(f"Progress: {completion_percentage:.2f}%  Translated: {i + 1}/{total_sentences}", end='\r')

    except KeyboardInterrupt:
        pass

    print("\nTranslation completed.")

    with open(dest_file, 'w', encoding='utf-8') as file:
        for i, sentence in enumerate(processed_sentences):
            
            file.write("<pl> " + sentence.strip() + "")
            if i < len(translated_sentences):
                file.write(" <fr> " + translated_sentences[i] + "")
            file.write('\n')

    print("Translated content saved to", dest_file)

if __name__ == "__main__":

    if len(sys.argv) != 7:
        print("Usage: python script.py <source_file> <dest_file> <source_lang> <source_lang_full> <target_lang> <repeat_count>")
        sys.exit(1)

    play(beep_sound2)
    
    source_file = sys.argv[1]
    dest_file = sys.argv[2]
    source_lang = sys.argv[3]
    source_lang_full = sys.argv[4]
    target_lang = sys.argv[5]
    repeat_count = int(sys.argv[6])
    
    #compare_transcription("się", source_lang, target_lang, 2)
    
    translate_to_french(source_file, dest_file, True, True, source_lang, source_lang_full, target_lang, repeat_count)
    
    
    
    # python translate.py test.html result.html pl polish fr 1
