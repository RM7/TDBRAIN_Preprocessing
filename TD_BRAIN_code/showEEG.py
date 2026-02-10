import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt

# --- CONFIGURATION ---
filename = '/Users/romaehler/Library/CloudStorage/OneDrive-HochschuleHeilbronn/Uni/Promotion/TDBRAIN_Datensatz/derivatives/preprocessed/sub-88032973/ses-1/eeg/sub-88032973_ses-1_task-restEC_eeg_csv_120.028s.npy'  # Change to your .npy file path
window_seconds = 30  # seconds to show at a time

# --- LOAD DATA ---
eeg = np.load(filename, allow_pickle=True)  # shape: (channels, samples) or (samples,) if single channel

sampling_rate = eeg['Fs']  # Hz, change to your EEG data's sampling rate
# If data shape is (1, 32, samples), squeeze the first dimension
eeg_data = np.squeeze(eeg['data'])  # shape: (32, samples)

channel_labels = eeg['labels']
artifacts = eeg['artifacts'] 

# Time axis for plotting
num_samples = eeg_data.shape[1]
duration = num_samples / sampling_rate
time = np.arange(num_samples) / sampling_rate

# Offset for each channel to stack them vertically
offset = -100  # Adjust this value based on your data's amplitude
offsets = np.arange(eeg_data.shape[0]) * offset

# Initial window
window_size = window_seconds
start_time = 0
end_time = start_time + window_size

fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(bottom=0.3)

lines = []
for i in range(eeg_data.shape[0]):
    (line,) = ax.plot([], [], label=channel_labels[i])
    lines.append(line)

ax.set_yticks(offsets)
ax.set_yticklabels(channel_labels)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Channels')

# --- Artifact colors ---
artifact_colors = {
    'VEOG': 'red',
    'HEOG': 'orange',
    'EMGtrl': 'blue',
    'JUMPtrl': 'cyan',
    'KURTtrl': 'lime',
    'SWINGtrl': 'pink',
    'EBtrl': 'purple',
}

# Slider for time navigation
axcolor = 'lightgoldenrodyellow'
ax_time = plt.axes([0.15, 0.08, 0.7, 0.02], facecolor=axcolor)
time_slider = Slider(
    ax=ax_time,
    label='Start Time (s)',
    valmin=0,
    valmax=max(0, duration - window_size),
    valinit=start_time,
    valstep=0.1,
)

# Slider for zoom (window size)
ax_zoom = plt.axes([0.15, 0.045, 0.7, 0.02], facecolor=axcolor)
zoom_slider = Slider(
    ax=ax_zoom,
    label='Window (s)',
    valmin=1,
    valmax=min(120, duration),
    valinit=window_size,
    valstep=0.1,
)

artifact_patches = []

def update(val):
    # Remove old artifact patches
    for patch in artifact_patches:
        patch.remove()
    artifact_patches.clear()

    start = time_slider.val
    window = zoom_slider.val
    end = min(start + window, duration)
    idx_start = int(start * sampling_rate)
    idx_end = int(end * sampling_rate)
    for i, line in enumerate(lines):
        data = eeg_data[i, idx_start:idx_end]
        # If channel 26 (index 26) and label contains 'artifact', scale from 0-1 to 0-100
        if i == 26 and 'artifact' in str(channel_labels[i]).lower():
            data = data * 50
        line.set_data(time[idx_start:idx_end], data + offsets[i])
    ax.set_xlim(start, end)
    ax.set_ylim(offsets[-1] + offset, offsets[0] - offset)

    # Draw artifact boxes for all channels
    for art_type, color in artifact_colors.items():
        if art_type in artifacts:
            for seg in artifacts[art_type]:
                seg_start, seg_end = seg
                # Only draw if in current window
                seg_start_s = seg_start / sampling_rate
                seg_end_s = seg_end / sampling_rate
                if seg_end_s < start or seg_start_s > end:
                    continue
                # Clip to window
                box_start = max(seg_start_s, start)
                box_end = min(seg_end_s, end)
                box_width = box_end - box_start
                # Draw box for all channels
                rect = Rectangle(
                    (box_start, offsets[-1] - offset),
                    box_width,
                    offsets[0] - offsets[-1],
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.2,
                    label=art_type
                )
                ax.add_patch(rect)
                artifact_patches.append(rect)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    # Only keep one handle per artifact type
    legend_dict = {}
    for h, l in zip(handles, labels):
        legend_dict[l] = h
    # Add artifact colors to legend
    for art_type, color in artifact_colors.items():
        legend_dict[art_type] = Rectangle((0,0),1,1,facecolor=color, edgecolor=color, alpha=0.5)
    ax.legend(legend_dict.values(), legend_dict.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

    fig.canvas.draw_idle()

time_slider.on_changed(update)
zoom_slider.on_changed(update)

# Initial plot
update(None)
plt.tight_layout()
plt.show()