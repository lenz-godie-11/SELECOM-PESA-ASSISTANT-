import speech_recognition as sr
from pydub import AudioSegment
import io

def transcribe_audio(wav_io):
    recognizer = sr.Recognizer()
    wav_io.seek(0)
    
    try:
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Samahani, sikuweza kuelewa sauti yako."
    except sr.RequestError as e:
        return f"Tatizo la mtandao: {e}"
    except Exception as e:
        return f"Tatizo la usindikaji: {e}"

def convert_ogg_to_wav(ogg_io):
    ogg_io.seek(0)
    audio = AudioSegment.from_file(ogg_io, format="ogg")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io