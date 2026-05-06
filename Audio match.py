from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sympy import fourier_series, pi, plot

sample_rate, song = wavfile.read("Song.wav")

if len(song.shape) == 2:
    mono_song = song.mean(axis=1).astype(song.dtype)
    
      
else:
    mono_song = song


total_samples = len(song)

time = np.arange(total_samples)/sample_rate

short_time = 4

samples_to_plot = int(short_time*sample_rate)



x_time = time[0:samples_to_plot]
y_song = mono_song[0:samples_to_plot]

plt.subplot(2,1,1)
plt.plot(x_time,y_song)
plt.title("Audio Signal in Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)




duration_t=15
start_t= 167
end_t = start_t + duration_t
query_start = int(start_t*sample_rate)
query_end = int(end_t*sample_rate)
xq_time = time[query_start:query_end]
y_query = mono_song[query_start:query_end]


plt.subplot(2,1,2)
plt.plot(xq_time,y_query)
plt.title("Query in time domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


def fourier_transform(array)

