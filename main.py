import math

import streamlit as st
import os
import nltk
import librosa
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pathlib import Path
from pydub import AudioSegment
from googletrans import Translator
from gtts  import gTTS
import errno, os, stat, shutil

def load_wav2vec_960h_model():
    """
  Returns the tokenizer and the model from pretrained tokenizers models
  """
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return tokenizer, model

def correct_uppercase_sentence(input_text):
    """
  Returns the corrected sentence
  """
    sentences = nltk.sent_tokenize(input_text)
    return (' '.join([s.replace(s[0], s[0].capitalize(), 1) for s in sentences]))

def asr_transcript(tokenizer, model, input_file):
    """
  Returns the transcript of the input audio recording

  Output: Transcribed text
  Input: Huggingface tokenizer, model and wav file
  """
    # read the file
    speech, samplerate = sf.read(input_file)
    # make it 1-D
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]
    # Resample to 16khz
    if samplerate != 16000:
        speech = librosa.resample(speech,orig_sr= samplerate,target_sr= 16000)
    # tokenize
    input_values = tokenizer(speech, return_tensors="pt").input_values
    # take logits
    logits = model(input_values).logits
    # take argmax (find most probable word id)
    predicted_ids = torch.argmax(logits, dim=-1)
    # get the words from the predicted word ids
    transcription = tokenizer.decode(predicted_ids[0])
    # output is all uppercase, make only the first letter in first word capitalized
    transcription = correct_uppercase_sentence(transcription.lower())
    return transcription

mypath = r"segments"
def smaller(Audio):
    if not os.path.isdir(mypath):
        print("Yes")
        os.makedirs(mypath)
    newAudios = []
    t1 = 0  # Works in milliseconds
    t2 = min(10 * 1000, len(Audio))
    print(len(Audio))
    a = 0
    for i in range(math.ceil(len(Audio) / 10000)):
        newAudio = Audio[t1:t2]
        newAudios.append(newAudio)
        t1 = t2
        t2 = min(t2 + (10 * 1000), len(Audio))
        newAudio.export(mypath + "/0" + "{}.wav".format(a), format="wav")
        a = a + 1

def delete_tmp(mypath):
    if os.path.isdir(mypath):
        os.remove(mypath)


def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

def run(path, arabic_text):
    Audio = AudioSegment.from_file(path)
    tokenizer, model = load_wav2vec_960h_model()
    smaller(Audio)
    try:
        os.mkdir("segments")
    except OSError as error:
        print(error)
    voices_names = os.listdir(mypath)
    voices_names = sorted(Path(mypath).iterdir(), key=os.path.getmtime)

    for i in voices_names:
        print(str(i) + ' : ')
        text = asr_transcript(tokenizer, model, str(i))
        arabic_text += Translator().translate(text, dest='ar').text
    shutil.rmtree("audio", ignore_errors=False, onerror=handleRemoveReadonly)
    shutil.rmtree("segments", ignore_errors=False, onerror=handleRemoveReadonly)
    return arabic_text
option = st.selectbox(
    'How would you like to upload ?',
    ('video and audio', 'only video', 'photo and audio',"photo and text"))

if option == 'video and audio' :
    uploaded_audio = st.file_uploader("Choose audio" , )
    uploaded_video= st.file_uploader("Choose video", type=["mp4" , "webm"])
    if uploaded_audio is not None:
        if uploaded_video is not None:

            audio_format = uploaded_audio.name.split(".")[1]
            video_format = uploaded_video.name.split(".")[1]
            try:
                os.mkdir("Wav2Lip/audio")
                os.mkdir("Wav2Lip/video")
            except OSError as error:
                print(error)

            os.chdir("Wav2Lip")
            with open("video/"+"temp."+video_format , "wb") as f:
                f.write(uploaded_video.getbuffer())

            try:
                os.mkdir("audio")
            except OSError as error:
                print(error)
            with open(os.path.join(r"audio/", "temp."+audio_format), "wb") as f:
                f.write(uploaded_audio.getbuffer())
            os.chdir("..")
            '''
            path = os.listdir("audio")[0]
            arabic_text = ""
            arabic_text = run("audio/" + path ,arabic_text )

            speech = gTTS(arabic_text, lang="ar")

            speech.save(r"Wav2Lip/audio/temp."+audio_format)
            
            st.text(arabic_text)
            '''
            os.chdir("Wav2Lip")
            os.system(f"python inference.py --checkpoint_path checkpoints/wav2lip.pth --face \"video/temp.{video_format}\" --audio \"audio/temp.{audio_format}\"")

            video_file = open('results/result_voice.mp4', 'rb')
            video_bytes = video_file.read()

            st.video(video_bytes)

            shutil.rmtree("audio", ignore_errors=False, onerror=handleRemoveReadonly)
            shutil.rmtree("video", ignore_errors=False, onerror=handleRemoveReadonly)
            os.chdir("..")

elif option == 'only video':
    uploaded_video = st.file_uploader("Choose video", type=["mp4"])
    if uploaded_video is not None:
        video_format = uploaded_video.name.split(".")[1]
        try:
            os.mkdir("Wav2Lip/audio")

        except OSError as error:
            print(error)
        try:
            os.mkdir("Wav2Lip/video")
        except OSError as error:
            print(error)
        os.chdir("Wav2Lip")
        with open("video/" + "temp." + video_format, "wb") as f:
            f.write(uploaded_video.getbuffer())
        os.chdir("..")
        try:
            os.mkdir("audio")
        except OSError as error:
            print(error)
        os.system("ffmpeg -y -i \"Wav2Lip/video/temp.mp4\" -f mp3 -ab 192000 \"audio/temp.wav\"")
        path = os.listdir("audio")[0]
        arabic_text = ""
        arabic_text = run("audio/" + path, arabic_text)
        speech = gTTS(arabic_text, lang="ar")
        speech.save("Wav2Lip/audio/temp.mp3")

        st.text(arabic_text)
        os.chdir("Wav2Lip")
        os.system(
            f"python inference.py --checkpoint_path checkpoints/wav2lip.pth --face \"video/temp.{video_format}\" --audio \"audio/temp.mp3\"")

        video_file = open('results/result_voice.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        shutil.rmtree("audio", ignore_errors=False, onerror=handleRemoveReadonly)
        shutil.rmtree("video", ignore_errors=False, onerror=handleRemoveReadonly)
        os.chdir("..")

elif option =='photo and audio':
    uploaded_photo = st.file_uploader("Choose photo" , type =['jpg', 'png', 'jpeg'])
    uploaded_audio = st.file_uploader("Choose audio", type=["mp3","wav","mpeg"])
    if uploaded_audio is not None:
        if uploaded_photo is not None:

            audio_format = uploaded_audio.name.split(".")[1]
            video_format = uploaded_photo.name.split(".")[1]
            try:
                os.mkdir("Wav2Lip/audio")
                os.mkdir("Wav2Lip/video")
            except OSError as error:
                print(error)

            os.chdir("Wav2Lip")
            with open("video/" + "temp." + video_format, "wb") as f:
                f.write(uploaded_photo.getbuffer())
            os.chdir("..")
            try:
                os.mkdir("audio")
            except OSError as error:
                print(error)
            with open(os.path.join(r"audio/", "temp." + audio_format), "wb") as f:
                f.write(uploaded_audio.getbuffer())
            path = os.listdir("audio")[0]
            arabic_text = ""
            arabic_text = run("audio/" + path, arabic_text)

            speech = gTTS(arabic_text, lang="ar")

            speech.save(r"Wav2Lip/audio/temp." + audio_format)

            st.text(arabic_text)
            os.chdir("Wav2Lip")
            os.system(
                f"python inference.py --checkpoint_path checkpoints/wav2lip.pth --face \"video/temp.{video_format}\" --audio \"audio/temp.{audio_format}\"")

            video_file = open('results/result_voice.mp4', 'rb')
            video_bytes = video_file.read()

            st.video(video_bytes)

            shutil.rmtree("audio", ignore_errors=False, onerror=handleRemoveReadonly)
            shutil.rmtree("video", ignore_errors=False, onerror=handleRemoveReadonly)
            os.chdir("..")

elif option =='photo and text':
    uploaded_photo = st.file_uploader("Choose photo" , type =['jpg', 'png', 'jpeg'])
    text = st.text_input("text")

    if text is not None:
        if uploaded_photo is not None:

            video_format = uploaded_photo.name.split(".")[1]
            try:
                os.mkdir("Wav2Lip/audio")
                os.mkdir("Wav2Lip/video")
            except OSError as error:
                print(error)

            os.chdir("Wav2Lip")
            with open("video/" + "temp." + video_format, "wb") as f:
                f.write(uploaded_photo.getbuffer())
            os.chdir("..")
            try:
                os.mkdir("audio")
            except OSError as error:
                print(error)


            arabic_text = text

            speech = gTTS(arabic_text, lang="ar")

            speech.save(r"Wav2Lip/audio/temp.mp3")

            st.text(arabic_text)
            os.chdir("Wav2Lip")
            os.system(
                f"python inference.py --checkpoint_path checkpoints/wav2lip.pth --face \"video/temp.{video_format}\" --audio \"audio/temp.mp3\"")

            video_file = open('results/result_voice.mp4', 'rb')
            video_bytes = video_file.read()

            st.video(video_bytes)
            shutil.rmtree("audio", ignore_errors=False, onerror=handleRemoveReadonly)
            shutil.rmtree("video", ignore_errors=False, onerror=handleRemoveReadonly)
            os.chdir("..")