import cv2
import numpy as np

# Load the image for the dog filter with transparency (4 channels: BGR and Alpha)
dog_ears = cv2.imread('spects.png', cv2.IMREAD_UNCHANGED)

# Load the face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if a frame was successfully read
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale for face detection, not for display
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Resize the dog ears to match the face size
        ears_width = w
        ears_height = int(ears_width * dog_ears.shape[0] / dog_ears.shape[1])
        ears_resized = cv2.resize(dog_ears, (ears_width, ears_height))

        # Calculate the position for the dog ears
        ears_y = y - int(ears_height / 2)
        ears_x = x

        # Overlay the dog ears
        if ears_y >= 0:
            alpha_ears = ears_resized[:, :, 3] / 255.0
            for c in range(0, 3):
                frame[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] = \
                    alpha_ears * ears_resized[:, :, c] + \
                    (1.0 - alpha_ears) * frame[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c]

    # Convert the entire frame (with ears overlay) to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the single channel grayscale image back to 3 channels for display
    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Display the resulting frame
    cv2.imshow('Dog Filter (Grayscale)', gray_frame_3ch)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
