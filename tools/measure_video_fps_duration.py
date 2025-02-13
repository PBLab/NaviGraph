import cv2

path_to_vid = '/home/elior/hdd/maze_master/apoE3_0_trimmed.mp4'

cap = cv2.VideoCapture(path_to_vid)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

print('fps = ' + str(fps))
print('number of frames = ' + str(frame_count))
print('duration (S) = ' + str(duration))
minutes = int(duration/60)
seconds = duration % 60
print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

cap.release()
