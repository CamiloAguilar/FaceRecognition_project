import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_profileface.xml')
mouth_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_mouth.xml')

if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

if profile_cascade.empty():
	raise IOError('No es posible cargar el clasificador xml para rostros laterales')

if mouth_cascade.empty():
	raise IOError('Error...')

cap = cv2.VideoCapture(0)
ds_factor = 1

while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,	interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#***********************
	## Rostros frontales
	#***********************
	front_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	for (x,y,w,h) in front_faces:
		# cara
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

		# ojos
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (x_eye,y_eye,w_eye,h_eye) in eyes:
			center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
			radius = int(0.3 * (w_eye + h_eye))
			color = (0, 255, 0)
			thickness = 1
			cv2.circle(roi_color, center, radius, color, thickness)

		# bocas
		mouths = mouth_cascade.detectMultiScale(gray, 1.7, 11)
		for (x,y,w,h) in mouths:
			y = int(y - 0.15*h)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

	#***********************
	## Rostros laterales
	#***********************
	prof_faces = profile_cascade.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in prof_faces:
		# cara
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

		# ojos
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (x_eye,y_eye,w_eye,h_eye) in eyes:
			center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
			radius = int(0.3 * (w_eye + h_eye))
			color = (0, 255, 0)
			thickness = 1
			cv2.circle(roi_color, center, radius, color, thickness)

		# bocas
		mouths = mouth_cascade.detectMultiScale(gray, 1.7, 11)
		for (x,y,w,h) in mouths:
			y = int(y - 0.15*h)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

	cv2.imshow('Detector de rostros', frame)
	# displaying the text on the screen
	img = ''
	cv2.putText(frame, 'OpenCV', (10,10), cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 2, cv2.LINE_AA)

	c = cv2.waitKey(1)
	if c == 0:
		break


cap.release()
cv2.destroyAllWindows()