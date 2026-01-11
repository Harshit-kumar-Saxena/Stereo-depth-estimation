import cv2

for i in range(8):
    cap = cv2.VideoCapture(i)
    print(i, cap.isOpened())
    cap.release()


# import cv2

# cap = cv2.VideoCapture(2)

# width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# print("Resolution:", int(width), "x", int(height))

# cap.release()


# import numpy as np

# image_width = 640  # Your camera width in pixels
# fov_degrees = 95.0  # From webcam specs
# fov_radians = fov_degrees * np.pi / 180

# focal_length_pixels = (image_width / 2) / np.tan(fov_radians / 2)
# print(f"Focal length: {focal_length_pixels:.2f} pixels")
