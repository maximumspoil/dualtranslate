import speech_recognition as sr

def compare_transcription(original_word, language, max_attempts):
    recognizer = sr.Recognizer()

    print(f"Say the word '{original_word}' in {language}:")
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")
        with sr.Microphone() as source:
            audio = recognizer.listen(source)

        try:
            recognized_text = recognizer.recognize_google(audio, language=language)
            print("You said:", recognized_text)

            if recognized_text.lower() == original_word.lower():
                print("Correct! You pronounced it correctly.")
                return  # Exit the loop if the word is pronounced correctly
            else:
                print("Incorrect. Your pronunciation differs from the original word.")

        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your speech.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    print(f"Maximum attempts reached ({max_attempts}). Moving on...")

# Example usage
word = "hello"
language = "en-US"  # Language code for English (United States)
max_attempts = 2
compare_transcription(word, language, max_attempts)