from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

"""the song file"""
sample_rate, song = wavfile.read("Song.wav")

"""no surround sound"""
if len(song.shape) == 2:
    mono_song = song.mean(axis=1).astype(song.dtype)   
else:
    mono_song = song


"""defining portion of signal"""
total_samples = len(song)
total_time = total_samples/sample_rate
time = np.arange(total_samples)/sample_rate
short_time = 4
samples_to_plot = int(short_time*sample_rate)
x_time = time[0:samples_to_plot]
y_song = mono_song[0:samples_to_plot]
"""plotting portion of signal"""
plt.subplot(2,1,1)
plt.plot(x_time,y_song)
plt.title("Audio Signal in Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

"""extracting the query clip"""
duration_t=15
start_t= 167
end_t = start_t + duration_t
query_start = int(start_t*sample_rate)
query_end = int(end_t*sample_rate)
xq_time = time[query_start:query_end]
y_query = mono_song[query_start:query_end]
"""plotting the query clip"""
plt.subplot(2,1,2)
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

"""plotting similarity score against time"""
plt.plot(times, scores)
plt.title("Similarity x time")
plt.xlabel("Time")
plt.ylabel("Similarity score")
plt.grid(True)
plt.show()
"""markers
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(times, scores, color="steelblue", linewidth=1.2, label="Similarity score")

# Actual clip position marker
ax.axvline(start_t, color="tab:green", linewidth=2, linestyle="--",
           label=f"Actual position  ({start_t} s)")
ax.annotate(f"Actual\n{start_t} s",
            xy=(start_t, scores[np.argmin(np.abs(times - start_t))]),
            xytext=(start_t + 5, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] else 0.9),
            arrowprops=dict(arrowstyle="->", color="tab:green"),
            color="tab:green", fontsize=9, fontweight="bold")

# Detected position marker
ax.axvline(matching_time, color="tab:red", linewidth=2, linestyle=":",
           label=f"Detected position ({matching_time:.1f} s)")
ax.annotate(f"Detected\n{matching_time:.1f} s",
            xy=(matching_time, best_score),
            xytext=(matching_time + 5, best_score * 0.92),
            arrowprops=dict(arrowstyle="->", color="tab:red"),
            color="tab:red", fontsize=9, fontweight="bold")

ax.set_title("Similarity Score vs Time  |  Actual & Detected Positions")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Cosine Similarity")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()"""

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

    




    
    
    

