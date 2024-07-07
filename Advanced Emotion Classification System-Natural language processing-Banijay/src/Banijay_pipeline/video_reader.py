import wave
import io

import detector
import numpy as np
import speech_recognition as sr
from googletrans import Translator
from moviepy.editor import VideoFileClip


def convert_to_wav(audio):
    shifted = audio.to_soundarray() * (2 ** 15 - 1)
    ints = shifted.astype(np.int16)

    with io.BytesIO() as wav_file:
        with wave.open(wav_file, 'wb') as wav_writer:
            wav_writer.setnchannels(1)
            wav_writer.setsampwidth(4)
            wav_writer.setframerate(audio.fps)

            wav_writer.writeframes(ints)

        wav_data = wav_file.getvalue()
    return wav_data


def transcribe_video_audio(video):
    translator = Translator()
    recognizer = sr.Recognizer()

    wav_data = convert_to_wav(video.audio)

    with sr.AudioFile(io.BytesIO(wav_data)) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language='nl-NL')
        text = translator.translate(text, src='nl', dest='en')
        return text.text

    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)
    except Exception as e:
        return "Could not request results; {0}".format(e)


def predict_video(model, tokenizer, input_video, episode, df, current_emotions):
    video = VideoFileClip(input_video)
    episode_df = df[df['Episode name'] == episode]

    data = {'predictions': [],
            'text': []}
    for index, row in episode_df.iterrows():
        start_time = row['Start Time (seconds)']
        end_time = row['End Time (seconds)']
        fragment = video.subclip(start_time, end_time)
        text = transcribe_video_audio(fragment)

        if "Could not request results" in text:
            text_all = ''

            steps = list(range(start_time, end_time, 100))
            for i in range(1, len(steps)):
                part1 = video.subclip(steps[i - 1], steps[i])
                current_text = transcribe_video_audio(part1)
                text_all += current_text

            text = text_all
        data['text'].append(text)
        data['predictions'].append(detector.predict(model, tokenizer, text, current_emotions))

    video.close()
    return data
