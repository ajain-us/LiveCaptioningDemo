import whisper
from multiprocessing import Process
import speech_recognition as sr
import os

def processingToAI(model, file_name, wav):
    output = 'output.txt'
    with open(file_name, 'wb') as f:
            f.write(wav)
    result = model.transcribe(file_name, language='english', fp16=False)
    print(result['text'])
    os.remove(file_name)
    with open(output, "a") as file:
        file.write(result['text'])
        file.write('\n')
    raise SystemExit


if __name__ == '__main__':
    r = sr.Recognizer()
    model = whisper.load_model("base")
    num = 0
    processes = []
    with sr.Microphone() as source:
        while(True):
            #r.adjust_for_ambient_noise(source)
            print('listening')
            audio = r.listen(source)
            print('encoding!')
            fileName = "testing" + str(num) + ".wav"
            num = num + 1
            Process(target=processingToAI, args=(model, fileName, audio.get_wav_data(),)).start()


