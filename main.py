import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from utils import preprocess_eye, load_eye_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load face detector and eye model
detector = MTCNN()
eye_model = load_eye_model()

# Parameters for drowsiness detection
closed_eye_frames = 0
threshold_frames = 15

# For MSE tracking
mse_values = []
frame_count = 0

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract eyes region
        keypoints = face['keypoints']
        left_eye = preprocess_eye(frame, keypoints['left_eye'])
        right_eye = preprocess_eye(frame, keypoints['right_eye'])

        # Predict eye state
        left_prediction = eye_model.predict(left_eye)
        right_prediction = eye_model.predict(right_eye)

        # Calculate MSE for each prediction (0: closed, 1: open)
        mse = ((left_prediction - 1)**2 + (right_prediction - 1)**2) / 2
        mse_values.append(mse[0][0])
        frame_count += 1

        # Determine if eyes are closed
        if left_prediction < 0.5 and right_prediction < 0.5:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        # Check if drowsy
        if closed_eye_frames >= threshold_frames:
            cv2.putText(frame, "Drowsy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot MSE values after the camera session ends
plt.plot(mse_values, label='MSE per frame')
plt.xlabel('Frame')
plt.ylabel('Mean Squared Error')
plt.title('Real-time MSE of Eye Predictions')
plt.legend()
plt.show()

# Evaluate model accuracy on testing data
test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_data = test_data_gen.flow_from_directory(
    r"C:\pythonProject\pythonProject\test",
    target_size=(24, 24),
    color_mode="grayscale",
    class_mode="binary",
    shuffle=False
)

# Get the testing accuracy
loss, accuracy = eye_model.evaluate(test_data)

