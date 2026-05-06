from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


sample_rate, song = wavfile.read("Song.wav")





def fourier_transform(signal):
    n = len(signal)
    result = fft(signal)
    magnitude = np.abs(result)
    one_sided = magnitude[:n//2]
    return one_sided



if len(song.shape) == 2:
    mono_song = song.mean(axis=1).astype(song.dtype)
    
      
else:
    mono_song = song


total_samples = len(song)
total_time = total_samples/sample_rate
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


query_fft = fourier_transform(y_query)
frequency = np.linspace(0, sample_rate/2,len(query_fft))

plt.plot(frequency, query_fft)
plt.title("Query in Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()



clip_length = len(y_query)
step_size = int(sample_rate * 0.5)
scores = []
times = []
for start in range(0, len(mono_song) - clip_length+1,step_size):
    window = mono_song[start:start+clip_length]
    window_fft = fourier_transform(window)
    similarity = np.dot(query_fft, window_fft)/ (np.linalg.norm(query_fft) * np.linalg.norm(window_fft)) 
    
    scores.append(similarity)
    times.append(start/sample_rate)
    

best_score_index = np.argmax(scores)
matching_time = times[best_score_index]
best_score = scores[best_score_index]

print(f"Sampling frequency:     {sample_rate} Hz")
print(f"Length of full signal:  {total_time} seconds")
print(f"Clip length:            {duration_t} seconds")
print(f"Original clip position: {start_t} seconds")
print(f"Detected position:      {matching_time} seconds")
print(f"Best similarity score:  {best_score}")

    




    
    
    

