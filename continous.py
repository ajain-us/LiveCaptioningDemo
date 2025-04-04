import whisper
import threading
import speech_recognition as sr
import os

r = sr.Recognizer()
model = whisper.load_model("base")
num = 0
output = 'output.txt'
threads = []

def processingToAI(model, file_name, wav):
    with open(fileName, 'wb') as f:
            f.write(wav)
    result = model.transcribe(file_name, language='english', fp16=False)
    print(result['text'])
    os.remove(file_name)
    with open(output, "a") as file:
        file.write(result['text'])
        file.write('\n')
    if('exit' in result['text']):
        raise SystemExit


with sr.Microphone() as source:
    while(True):
        #r.adjust_for_ambient_noise(source)
        print('listening')
        audio = r.listen(source)
        print('encoding!')
        fileName = "testing" + str(num) + ".wav"
        num = num + 1
        threading.Thread(target=processingToAI(model, fileName, audio.get_wav_data())).start()


