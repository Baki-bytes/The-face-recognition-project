import face_recognition
import cmake
import numpy as np
import cv2
import csv
from datetime import datetime


video_captur = cv2.VideoCapture(0)

# load known faces

salman_image = face_recognition.load_image_file("faces/image 1.jpeg")
salman_encoding = face_recognition.face_encodings(salman_image)[0]
# here we write 0 because face_encodings returns the lists of the faces in that particular image
# so in this code we just pass a single image in a single photo.

shahrukh_image = face_recognition.load_image_file("faces/image 2.jpeg")
shahrukh_encoding = face_recognition.face_encodings(shahrukh_image)[0]
# now we need to make face encodings :- encodings means to convert images into a number such that it is easier to compare

known_face_encoding = [salman_encoding, shahrukh_encoding]k
known_face_names = ["Salman Khan", "Shah rukh khan"]

# list of expected students
Students = known_face_names.copy()
face_locations = []
face_encodings = []

# get the current date and time :-
now = datetime.now()
current_date = now.strftime("%y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
line_writer = csv.writer(f)

while True:
    (
        _,
        frame,
    ) = (
        video_captur.read()
    )  # we write here underscore because the first arrgument of  that wheather your video capture is successful or not ? video_captur.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces: -
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encodings in face_encodings:
        match = face_recognition.compare_faces(known_face_encoding, face_encodings)
        face_Distance = face_recognition.face_distance(
            known_face_encoding, face_encodings
        )
        best_match_index = np.argmin(face_Distance)

        if match[best_match_index]:
            name = known_face_names[best_match_index]
            cv2.imshow("Attendace", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
