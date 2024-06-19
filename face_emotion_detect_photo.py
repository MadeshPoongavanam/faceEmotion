import matplotlib.pyplot as plt
from fer import FER
import cv2

# Load the image
image_path = 'photo_emotions2.jpeg'
image = cv2.imread(image_path)

# Convert the image to RGB (required by fer library)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the FER detector
detector = FER(mtcnn=True)

# Detect emotions in the image
results = detector.detect_emotions(image_rgb)

# Print the results
for result in results:
    print(f"Box: {result['box']}")
    print(f"Emotions: {result['emotions']}")

# Plot the image with detected emotions
fig, ax = plt.subplots()
ax.imshow(image_rgb)
for result in results:
    (x, y, w, h) = result['box']
    emotions = result['emotions']
    dominant_emotion = max(emotions, key=emotions.get)
    ax.text(x, y, dominant_emotion, color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    rect = plt.Rectangle((x, y), w, h, fill=False, color='blue')
    ax.add_patch(rect)

plt.axis('off')
plt.show()
