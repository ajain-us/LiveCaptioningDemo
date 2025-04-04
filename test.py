import whisper
import sounddevice as sd
from scipy.io.wavfile import write

freq = 44100
duration = 5
sd.default.channels = 1

print("please start speaking")
recording = sd.rec(int(duration * freq), samplerate=freq)
sd.wait()
print("recording done")

print("playing back and decoding")
sd.play(recording, freq)
sd.wait()

write("test.wav", freq, recording)

model = whisper.load_model("base")

result = model.transcribe("test.wav", verbose = True)
print(result['text'])