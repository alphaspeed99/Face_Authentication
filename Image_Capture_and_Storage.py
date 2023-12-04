import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

data_dir = "data"
encoder = LabelEncoder()

def preprocess_image(image):
    # Preprocess the image as needed (e.g., resize, convert to grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))
    return resized_image

# Create directories to store images of individuals
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow("Capture Images", frame)

    # Capture images when 'c' is pressed
    key = cv2.waitKey(1)
    if key == ord('c'):
        name = input("Enter the person's name: ")
        person_dir = os.path.join(data_dir, name)
        if not os.path.exists(person_dir):
            os.mkdir(person_dir)

        image_path = os.path.join(person_dir, f"{len(os.listdir(person_dir)) + 1}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")

    # Exit capturing when 'q' is pressed
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Load images and labels
data = []
labels = []

for person_name in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person_name)
    if os.path.isdir(person_dir):
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            preprocessed_image = preprocess_image(image)
            data.append(preprocessed_image)
            labels.append(person_name)

# Extract features from data and labels
data_features = np.array(data).reshape(len(data), -1)
labels_encoded = encoder.fit_transform(labels)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(data_features, labels_encoded)

# Save the trained model and encoder classes
joblib.dump(svm_model, 'svm_model.pkl')
np.save('encoder_classes.npy', encoder.classes_)
