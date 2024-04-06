from flask import Flask, render_template, request, jsonify, session
import io
import sys
import os
import cv2
import numpy as np
import mysql.connector
import datetime
from flask_cors import CORS
from flask_sslify import SSLify
import base64
import numpy as np
import io
import cv2

app = Flask(__name__)
sslify = SSLify(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = os.urandom(24)

# Connect to MySQL database and define functions for updating attendance and recognizing faces
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="attendance_db"
)
cursor = db.cursor()

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def update_attendance(student_id, username='default_username'):
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    query = f"INSERT INTO attendance (student_id, username, status, capture_time) VALUES ({student_id}, '{username}', 'present', '{current_datetime}')"
    cursor.execute(query)
    db.commit()

def get_username(student_id):
    query = f"SELECT username FROM students WHERE id = {student_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return 'Unknown'

def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 1:
        print("Multiple faces detected. Please align only one face at a time.")
        return 'Unknown'
    if len(faces) == 0:
        return 'No face detected'

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)
        
        if confidence < 70:
            username = get_username(id_)
            session_username = session.get('username')
            print(f"Recognized Username: {username}, Session Username: {session_username}")
            if username == session_username:
                update_attendance(id_, username)
                print(f"Attendance marked for student ID: {id_}, Username: {username}")
                return username
        
    return 'Unknown'

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        session['username'] = username  # Store session username
        image_data = data.get('image')
        print("Received username:", username)
        print("Received image data:", image_data[:100])
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)

        recognized_username = recognize_faces(frame)

        if recognized_username != 'Unknown':
            print("Attendance marked successfully.")
            return jsonify({'success': True, 'message': 'Attendance marked successfully'})
        else:
            print("No face detected or face not recognized or username mismatch.")
            return jsonify({'success': False, 'message': 'No face detected or face not recognized or username mismatch'})
    else:
        return jsonify({'success': False, 'message': 'Method not allowed'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, ssl_context=('localhost.pem', 'localhost-key.pem'))
