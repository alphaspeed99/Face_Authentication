import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from subprocess import call


# Load the pre-trained SVM model and LabelEncoder
svm_model = SVC(kernel='linear')
svm_model = joblib.load('dataset/svm_model.pkl')

encoder = LabelEncoder()
encoder.classes_ = np.load('dataset/encoder_classes.npy')

# Initialize camera
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]

        # Resize the face_roi to match the expected input size
        resized_face_roi = cv2.resize(face_roi, (64, 64))

        # Flatten the 2D grayscale image into a 1D array
        flattened_face_roi = resized_face_roi.flatten()

        predicted_label = svm_model.predict([flattened_face_roi])
        predicted_person = encoder.inverse_transform([predicted_label])[0]

        # Display the identified person's name
        cv2.putText(frame, f"Identified: {predicted_person}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Perform further processes based on the identified person
        if predicted_person == "alpha":
            cv2.putText(frame, "Welcome, alpha!", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Authentication", frame)
            cv2.waitKey(2000)  # Show the "Welcome" message for 2 seconds
            cap.release()  # Release the camera resources
            cv2.destroyAllWindows()


            # Run another Python script
            def open_py_file():
                call(["python","main.py"])

            open_py_file()
            exit()
        elif predicted_person == "pk":
            cv2.putText(frame, "Hello, pk!", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Authentication", frame)
            cv2.waitKey(2000)  # Show the "Welcome" message for 2 seconds
            cap.release()  # Release the camera resources
            cv2.destroyAllWindows()


            # Run another Python script
            def open_py_file():
                call(["python","main.py"])

            open_py_file()
            exit()
        else:
            cv2.putText(frame, "Unknown Person", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Handle unknown person

    cv2.imshow("Face Authentication", frame)

    # Exit real-time authentication when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
