import cv2
import face_recognition

imgHadi = face_recognition.load_image_file('image/hadi.jpg')
imgHadi = cv2.cvtColor(imgHadi, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('image/Saim.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgHadi)[0]
encodeHadi = face_recognition.face_encodings(imgHadi)[0]
cv2.rectangle(imgHadi, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255),2 )

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255),2 )

results = face_recognition.compare_faces([encodeHadi], encodeTest)
facDis = face_recognition.face_distance([encodeHadi],encodeTest)
print(results, facDis)

cv2.imshow('Abdul hadi', imgHadi)
cv2.imshow('Abdul test', imgTest)
cv2.waitKey(0)
