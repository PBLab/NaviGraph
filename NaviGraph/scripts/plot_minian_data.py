from typing import Union
import os
import xarray as xr
import dask.array as darr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib


def open_minian(dpath: str, return_dict=False) -> Union[dict, xr.Dataset]:
    if os.path.isfile(dpath):
        ds = xr.open_dataset(dpath).chunk()
    elif os.path.isdir(dpath):
        dslist = []
        for d in os.listdir(dpath):
            arr_path = os.path.join(dpath, d)
            if os.path.isdir(arr_path):
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(os.path.join(arr_path, arr.name), inline_array=True)
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    return ds


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


def rgba_to_rgb(rgba):
    return tuple(int(255*i) for i in rgba[:3])


# Constants
SHOW = True
SAVE = True

# Filter by max peak
FILTER_BY_MAX_PEAK = True
max_peak = 20

# Styling
plt.style.use("fivethirtyeight")
# cmap = matplotlib.cm.get_cmap('plasma') # choose this for sequential colormap
cmap = matplotlib.cm.get_cmap('Paired').colors  # choose this for qualitative colormap

# Paths
minian_path = '/home/elior/hdd/maze_master/processed_minain_demo/minian_zarr'
video_path = '/home/elior/hdd/maze_master/processed_minain_demo/video/0.avi'
export_path = '/home/elior/hdd/maze_master/processed_minain_demo/export'

# Choose units to display
unit_id_to_display = np.arange(0, -1)

# Load data
ds = open_minian(minian_path)
if len(unit_id_to_display) == 0:
    unit_id_to_display = ds.A.unit_id.values

# Spatial_data
spatial_data_grouped_by_unit_id = ds.A.groupby('unit_id')

# Temporal_data
temporal_data_grouped_by_unit_id = ds.C.groupby('unit_id')
time_series_data = np.array([temporal_data_grouped_by_unit_id[unit_id].values for unit_id in unit_id_to_display]).squeeze()
time_series_df = pd.DataFrame(time_series_data.T, columns=unit_id_to_display)

if FILTER_BY_MAX_PEAK:
    filtered_unit_id = time_series_df.max()[(time_series_df.max() > max_peak)].index.values
    time_series_df = time_series_df[filtered_unit_id]
    unit_id_to_display = filtered_unit_id

# Set colors
# unit_id_color = [rgba_to_rgb(cmap(np.random.random())) for _ in range(len(unit_id_to_display))]  # choose this for sequential colormap
unit_id_color = [rgba_to_rgb(cmap[np.random.randint(0, len(cmap))]) for _ in range(len(unit_id_to_display))]  # choose this for qualitative colormap

plt_w, plt_h = 20, 7
fig = time_series_df.plot(figsize=(plt_w, plt_h), color=list(map(rgb_to_hex, unit_id_color)), lw=1.5)

plt.ion()

# time plots
time_bar_plots = fig.axvline(x=0, color='#8B0000', linestyle='--', linewidth=2)

fig.set_xlabel("Frames")
fig.set_ylabel("Unit Ca Intensity [a.u.]")
plt.suptitle("Neuronal Traces")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
video_writer = cv2.VideoWriter(filename=os.path.join(export_path, 'processed_video.mp4'),
                               fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                               fps=fps,
                               frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
plot_writer = cv2.VideoWriter(filename=os.path.join(export_path, 'trace.mp4'),
                              fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                              fps=fps,
                              frameSize=(int(plt_w * 100), int(plt_h * 100)))
frame_count = -1
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        cap.release()
        cv2.destroyAllWindows()
        break

    frame_count += 1

    # draw frame mark on timeseries plot
    time_bar_plots.set_xdata(frame_count)
    fig.figure.canvas.draw()
    fig.figure.canvas.flush_events()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = np.array(Image.open(img_buf))

    # draw unit_id_mask on frame
    for ind, unit_id in enumerate(unit_id_to_display):
        unit_id_mask = (spatial_data_grouped_by_unit_id[unit_id].values.squeeze() > 0.).astype(np.uint8)
        unit_id_contours, _ = cv2.findContours(unit_id_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, unit_id_contours, -1, unit_id_color[ind][::-1], 1)

    cv2.putText(frame, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    if SHOW:
        cv2.imshow('frame', frame)

    if SAVE:
        video_writer.write(frame)
        plot_writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
plot_writer.release()
cv2.destroyAllWindows()

