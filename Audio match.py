from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

sample_rate, song = wavfile.read("Song.wav")

if len(song.shape) == 2:
    mono_song = song.mean(axis=1).astype(song.dtype)
    
      
else:
    mono_song = song


total_samples = len(song)

time = np.arange(total_samples)/sample_rate


short_time = 10
samples_to_plot = int(short_time*sample_rate)


x_time = time[0:samples_to_plot]
y_song = mono_song[0:samples_to_plot]

plt.plot(x_time,y_song)
plt.title("Audio Signal in Time Domain (First 50ms)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

 