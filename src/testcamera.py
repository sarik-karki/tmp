import cv2
import time

print("Testing ELP Camera (Overhead)...")
cap_elp = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Force MJPEG codec and HD resolution
cap_elp.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_elp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap_elp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

# ---> NEW: Let the camera "warm up" so the Auto White Balance fixes the yellow tint
print("Warming up ELP sensor...")
for _ in range(30):
    cap_elp.read()
    time.sleep(0.01)
# --------------------------------------------------------------------------------

ret1, frame1 = cap_elp.read()
if ret1:
   cv2.imwrite("ELP_overhead_test.jpg", frame1)
   print("Saved ELP_overhead_test.jpg")
else:
   print("Failed to grab frame from ELP Camera.")
cap_elp.release()


print("\nTesting NexiGo N60 (Entrance)...")
cap_nexigo = cv2.VideoCapture(2, cv2.CAP_V4L2)

# Force MJPEG codec and HD resolution
cap_nexigo.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_nexigo.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_nexigo.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---> NEW: Warm up the NexiGo sensor too
print("Warming up NexiGo sensor...")
for _ in range(30):
    cap_nexigo.read()
    time.sleep(0.01)
# ----------------------------------------

ret2, frame2 = cap_nexigo.read()
if ret2:
   cv2.imwrite("NexiGo_entrance_test.jpg", frame2)
   print("Saved NexiGo_entrance_test.jpg")
else:
   print("Failed to grab frame from NexiGo Camera.")
cap_nexigo.release()

print("\nCamera test complete!")