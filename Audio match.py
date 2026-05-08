from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


sample_rate, song = wavfile.read("Song.wav")


if len(song.shape) == 2:
    mono_song = song.mean(axis=1).astype(song.dtype)   
else:
    mono_song = song


"""defining portion of signal"""
total_samples = len(song)
total_time = total_samples/sample_rate
time = np.arange(total_samples)/sample_rate
short_time = 15
samples_to_plot = int(short_time*sample_rate)
x_time = time[0:samples_to_plot]
y_song = mono_song[0:samples_to_plot]
"""plotting portion of signal"""

plt.plot(x_time,y_song)
plt.title("Audio Signal in Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
"""extracting the query clip"""
duration_t=15
start_t= 167
end_t = start_t + duration_t
query_start = int(start_t*sample_rate)
query_end = int(end_t*sample_rate)
xq_time = time[query_start:query_end]
y_query = mono_song[query_start:query_end]
"""plotting the query clip"""

plt.plot(xq_time,y_query)
plt.title("Query in time domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

"""defining a method to turn signal to f domain"""
def fourier_transform(signal):
    n = len(signal)
    result = fft(signal)
    magnitude = np.abs(result)
    one_sided = magnitude[:n//2]
    return one_sided

"""apply function to query clip"""
query_fft = fourier_transform(y_query)
frequency = np.linspace(0, sample_rate/2,len(query_fft))
"""plotting new f domain query clip"""
plt.plot(frequency, query_fft)
plt.title("Query in Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

"""sliding window algorithm for audio matching"""
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


times_array = np.array(times)
actual_time_index = np.argmin(np.abs(times_array - start_t))
actual_score = scores[actual_time_index]

# Plot the ACTUAL position marker ( a green circle 'o')

plt.plot(start_t, actual_score, marker='o', markersize=10, color="tab:green", linestyle='None', label=f"Actual Position ({start_t} s)")
plt.annotate(f"Actual: {start_t} s", 
             xy=(start_t, actual_score), 
             xytext=(10, -15), textcoords="offset points", 
             color="tab:green", fontweight="bold", ha = "right")
# Plot the DETECTED position marker ( a red star '*')
plt.plot(matching_time, best_score, marker='*', markersize=15, color="tab:red", linestyle='None', label=f"Detected Position ({matching_time:.1f} s)")
plt.annotate(f"Detected: {matching_time:.1f} s", 
             xy=(matching_time, best_score), 
             xytext=(10, -15), textcoords="offset points", 
             color="tab:red", fontweight="bold", ha = "left")

"""plotting similarity score against time"""
plt.plot(times, scores)
plt.title("Similarity score vs time")
plt.xlabel("Time")
plt.ylabel("Similarity score")
plt.grid(True)
plt.show()


"""original vs detected"""
det_start  = int(matching_time  * sample_rate)
det_end    = det_start + clip_length
y_detected = mono_song[det_start:det_end]
xd_time    = np.linspace(matching_time, matching_time + duration_t, len(y_detected))

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

# Original
axes[0].plot(xq_time, y_query, color="tab:orange", linewidth=0.8)
axes[0].set_title("Original Query Clip")
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.4)

# Detected
axes[1].plot(xd_time, y_detected, color="tab:red", linewidth=0.8)
axes[1].set_title("Detected Matching Segment")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.4)

plt.tight_layout(); plt.show()

print(f"Sampling frequency:     {sample_rate} Hz")
print(f"Length of full signal:  {total_time} seconds")
print(f"Clip length:            {duration_t} seconds")
print(f"Original clip position: {start_t} seconds")
print(f"Detected position:      {matching_time} seconds")
print(f"Best similarity score:  {best_score}")

    




    
    
    

