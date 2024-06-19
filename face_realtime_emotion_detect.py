import cv2
from fer import FER
import numpy as np
# from datetime import datetime

# Initialize the FER emotion detector with MTCNN for face detection
detector = FER(mtcnn=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

def draw_text(frame, text, x, y):
    """Draw text with background."""
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x
    text_y = y - text_size[1]
    cv2.rectangle(frame, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (text_x, y), font, font_scale, (255, 255, 255), font_thickness)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB (FER library requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotions in the frame
    result = detector.detect_emotions(rgb_frame)

    # Loop through detected faces
    for face in result:
        (x, y, w, h) = face['box']
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the dominant emotion
        dominant_emotion = max(face['emotions'], key=face['emotions'].get)
        score = face['emotions'][dominant_emotion]

        # Display the emotion on the frame
        draw_text(frame, f'{dominant_emotion} ({score:.2f})', x, y - 10)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
