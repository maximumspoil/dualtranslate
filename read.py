import sys
import re
import os
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import time

# I had a similar problem (the same kind of "PermissionError: [Errno 13] Permission denied") with playback on Windows 8.1, and installing simpleaudio (executing 'pip install simpleaudio') solved the issue.
def read_text(text, language):
    tts = gTTS(text, lang=language)  
    print(f"Reading ({language}): {text}")
    tts.save("temp.mp3")
    audio = AudioSegment.from_mp3("temp.mp3")
    play(audio)

def read_text_with_language(text, start_sentence):
    # Separate the text into sentences based on language tags
    sentences = []

    # Use regex to find language tags and sentences
    pattern = r"(?P<language><pl>|<fr>)(?P<sentence>.*?)(?=(?:<pl>|<fr>))"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        language = match.group("language")
        sentence = match.group("sentence").strip()
        if sentence:
            sentences.append((language, sentence))

    # Adjust start_sentence to be zero-based index
    start_sentence -= 1

    # Set the starting sentence based on the argument start_sentence
    sentences = sentences[:start_sentence]


    try:
        # Read each sentence using the appropriate voice
        for language, sentence in sentences:
            if language == "<pl>":
                read_text(sentence, 'pl')
            elif language == "<fr>":
                read_text(sentence, 'fr')
            else:
                continue

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping the text-to-speech.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <text_file_path> <start_sentence>")
        sys.exit(1)

    text_file_path = sys.argv[1]
    start_sentence = int(sys.argv[2])

    with open(text_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    read_text_with_language(text, start_sentence)
