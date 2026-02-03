from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify, send_from_directory
import sqlite3
import os
import subprocess
import shutil
import requests
import cv2
import json
import smtplib
import time
from email.message import EmailMessage
from werkzeug.utils import secure_filename
from datetime import datetime
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from facerecognition import recognize_face_from_frame
from lib.face import detect_single  # Make sure this is imported
import numpy as np
import mediapipe as mp
from collections import deque
last_processed_time = 0
from datetime import datetime
import pytz
recognition_buffer = deque(maxlen=3)
import paho.mqtt.client as mqtt
from flask import Response
import atexit
MQTT_BROKER = "konnect.robosap.co.in"
MQTT_PORT = 1883
MQTT_USER = "robosaptwo"
MQTT_PASSWORD = "337adxl2023"
MQTT_PUB_TOPIC = "faceticket"
MQTT_SMS_TOPIC = "robosms"
MQTT_SUB_TOPIC = "faceticket2"

mqtt_client = mqtt.Client()

# Enable/Disable Fraud Email Alerts
EMAIL_ALERTS_ENABLED = False # Set to False to disable sending fraud alert emails
SMS_ALERTS_ENABLED = False

# Authenticate
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

# Connect to the broker
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Optional: subscribe to listen for messages from 'doorbell'
def on_message(client, userdata, msg):
    print(f"- Received from {msg.topic}: {msg.payload.decode()}")

mqtt_client.on_message = on_message
mqtt_client.subscribe(MQTT_SUB_TOPIC)
mqtt_client.loop_start()  # Start loop in background
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
EYE_AR_THRESHOLD = 2
EYE_CLOSED_DURATION_REQUIRED = 3
LIVENESS_TEST_DURATION = 6

liveness_timer_start = 0
eye_closed_total = 0
eye_closed_start = None

camera = cv2.VideoCapture(0)  # (Global camera access)
if not camera.isOpened():
    raise Exception("- Could not open video device")

print("- Camera initialized")

# - Ensure necessary tables exist
def init_database():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create admin table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    ''')

    # Create students table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        usn TEXT UNIQUE,
        branch TEXT,
        semester TEXT,
        hallticket TEXT UNIQUE,
        block TEXT
    )
    ''')

    # Create attendance table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usn TEXT,
        timestamp TEXT
    )
    ''')

    # Create fraud table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fraud (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        timestamp TEXT
    )
    ''')

    # Insert default admin if table is empty
    cursor.execute('SELECT COUNT(*) FROM admin')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO admin (username, password) VALUES (?, ?)', ('admin', 'admin123'))
        print("- Default admin account created: username=admin, password=admin123")

    conn.commit()
    conn.close()
    print("- Database initialized successfully!")

# ✨ Call it immediately
init_database()
app = Flask(__name__)
app.secret_key = 'your_secret_key'

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')

# Liveness model
liveness_model = load_model("model.h5")
liveness_model.summary()


# Frame-based verification buffer
face_status_buffer = []
MAX_FRAME_BUFFER = 10
fraud_delay_active = False
last_fraud_time = 0
FRAUD_DELAY_SECONDS = 3

# Email cooldown
last_fraud_email_time = 0
EMAIL_COOLDOWN_SECONDS = 5


@atexit.register
def cleanup():
    print("- Releasing Camera...")
    camera.release()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in HTTP multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def publish_mqtt(status,message):
    payload = json.dumps({"status": status, "message": message})
    mqtt_client.publish(MQTT_PUB_TOPIC, payload)
    print(f"- Published to {MQTT_PUB_TOPIC}: {payload}")


def publish_mqttsms(number):
    payload = json.dumps({"number": number, "msg": "Unauthorized Person Detected"})
    mqtt_client.publish(MQTT_SMS_TOPIC, payload)
    print(f"- Published to {MQTT_SMS_TOPIC}: {payload}")

def eye_ratio(landmarks, image_shape):
    ih, iw = image_shape[:2]
    def dist(p1, p2):
        x1, y1 = int(landmarks[p1].x * iw), int(landmarks[p1].y * ih)
        x2, y2 = int(landmarks[p2].x * iw), int(landmarks[p2].y * ih)
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Landmarks for both eyes
    left = dist(385, 380) + dist(387, 373)
    left_w = dist(362, 263)
    right = dist(160, 144) + dist(158, 153)
    right_w = dist(33, 133)
    return (left / (2.0 * left_w) + right / (2.0 * right_w)) / 2



def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def send_fraud_email(image_path):
    global last_fraud_email_time

    if not EMAIL_ALERTS_ENABLED:
        print(" Email alerts disabled by flag.")
        return

    now = time.time()
    if now - last_fraud_email_time < EMAIL_COOLDOWN_SECONDS:
        print(" Skipping fraud email due to cooldown.")
        return
    last_fraud_email_time = now

    EMAIL_ADDRESS = 'mailsendfromuser@gmail.com'
    EMAIL_PASSWORD = 'rjmobvnskfmtxcjy'

    msg = EmailMessage()
    msg['Subject'] = ' Fraud Detected - Face Ticket System'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = 'pruthvirajpandav4777@gmail.com'
    msg.set_content('A fraud attempt was detected. Image attached.')

    try:
        with open(image_path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("- Fraud email sent successfully!")
    except Exception as e:
        print(f"- Failed to send fraud email: {e}")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    name = request.form.get('name')
    usn = request.form.get('usn')

    if not name or not usn:
        return jsonify({'status': 'error', 'message': 'Missing name or USN'})

    folder_name = f"{name.replace(' ', '_')}_{usn}"
    student_dir = os.path.join('training_data', folder_name)
    os.makedirs(student_dir, exist_ok=True)

    success, frame = camera.read()
    if not success:
        return jsonify({'status': 'error', 'message': 'Camera read failed'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_box = detect_single(gray)

    if face_box is None:
        return jsonify({'status': 'noface'})

    files = os.listdir(student_dir)
    count = len([f for f in files if f.endswith('.jpg')])
    filename = f"{count:03d}.jpg"
    path = os.path.join(student_dir, filename)
    cv2.imwrite(path, frame)

    return jsonify({'status': 'success'})


@app.route('/admin_login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM admin WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['admin'] = username
            return redirect('/dashboard')
        else:
            flash('Invalid credentials!', 'danger')
    return render_template('login.html')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    name = request.form.get('name')
    usn = request.form.get('usn')
    if not name or not usn:
        return {'status': 'error', 'message': 'Missing name or USN'}

    folder_name = f"{name.replace(' ', '_')}_{usn}"
    student_dir = os.path.join('training_data', folder_name)
    os.makedirs(student_dir, exist_ok=True)

    image = request.files['image']
    files = os.listdir(student_dir)
    count = len([f for f in files if f.endswith('.jpg')])
    filename = f"{count:03d}.jpg"
    path = os.path.join(student_dir, filename)
    image.save(path)

    # - Validate that a face is actually in the image before keeping it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if detect_single(gray) is None:
        os.remove(path)
        return {'status': 'noface'}

    return {'status': 'success'}

@app.route('/dashboard')
def dashboard():
    if 'admin' not in session:
        return redirect('/admin_login')
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect('/admin_login')

@app.route('/train')
def train_model():
    if 'admin' not in session:
        return redirect('/admin_login')

    import sys
    try:
        result = subprocess.run(
            [sys.executable, 'train.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()  # ensure correct working directory
        )

        print(" STDOUT:\n", result.stdout)
        print("- STDERR:\n", result.stderr)

        if result.returncode == 0:
            flash('- Training completed!', 'success')
            flash(f"<pre>{result.stdout}</pre>", 'info')
        else:
            flash('- Training failed!', 'danger')
            flash(f"<pre>{result.stderr}</pre>", 'danger')

    except Exception as e:
        print(f"- Training Error: {str(e)}")
        flash(f'- Exception during training: {str(e)}', 'danger')

    return redirect('/dashboard')


@app.route('/students')
def view_students():
    if 'admin' not in session:
        return redirect('/')
    conn = get_db_connection()
    students = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    return render_template('students.html', students=students)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'admin' not in session:
        return redirect('/admin_login')

    if request.method == 'POST':
        name = request.form['name']
        usn = request.form['usn']
        branch = request.form['branch']
        semester = request.form['semester']
        hallticket = request.form['hallticket']
        block = request.form['block']

        student_dir = f"training_data/{name.replace(' ', '_')}_{usn}"
        os.makedirs(student_dir, exist_ok=True)

        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO students (name, usn, branch, semester, hallticket, block)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, usn, branch, semester, hallticket, block))
            conn.commit()
            flash('- Student registered successfully!', 'success')
            return redirect('/dashboard')

        except sqlite3.IntegrityError as e:
            if os.path.exists(student_dir):
                shutil.rmtree(student_dir)

            if 'usn' in str(e):
                flash('- Error: A student with this USN already exists.', 'danger')
            elif 'hallticket' in str(e):
                flash('- Error: A student with this Hall Ticket number already exists.', 'danger')
            else:
                flash(f'- Database error: {str(e)}', 'danger')

        except Exception as e:
            if os.path.exists(student_dir):
                shutil.rmtree(student_dir)
            flash(f'- Unexpected error: {str(e)}', 'danger')

        finally:
            conn.close()

    return render_template('register.html')

@app.route('/edit_student/<int:id>', methods=['GET', 'POST'])
def edit_student(id):
    if 'admin' not in session:
        return redirect('/')
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE id = ?', (id,)).fetchone()
    if request.method == 'POST':
        name = request.form['name']
        usn = request.form['usn']
        branch = request.form['branch']
        semester = request.form['semester']
        hallticket = request.form['hallticket']
        block = request.form['block']
        conn.execute('UPDATE students SET name=?, usn=?, branch=?, semester=?, hallticket=?, block=? WHERE id=?',
                     (name, usn, branch, semester, hallticket, block, id))
        conn.commit()
        conn.close()
        flash('- Student updated!', 'success')
        return redirect('/students')
    conn.close()
    return render_template('edit_student.html', student=student)

@app.route('/delete_student/<int:id>')
def delete_student(id):
    if 'admin' not in session:
        return redirect('/')
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE id = ?', (id,)).fetchone()
    if student:
        folder = os.path.join('training_data', f"{student['name'].replace(' ', '_')}_{student['usn']}")
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                flash(f"- Could not delete image folder: {e}", "danger")
        conn.execute('DELETE FROM students WHERE id = ?', (id,))
        conn.commit()
        flash(' Student deleted.', 'info')
    else:
        flash("- Student not found.", "danger")
    conn.close()
    return redirect('/students')

@app.route('/attendance')
def view_attendance():
    if 'admin' not in session:
        return redirect('/')
    page = int(request.args.get('page', 1))
    per_page = 10
    offset = (page - 1) * per_page
    conn = get_db_connection()
    records = conn.execute('''
        SELECT a.id, a.timestamp, s.name, s.usn, s.block
        FROM attendance a
        LEFT JOIN students s ON a.usn = s.usn
        ORDER BY a.timestamp DESC
        LIMIT ? OFFSET ?
    ''', (per_page, offset)).fetchall()
    total_count = conn.execute('SELECT COUNT(*) FROM attendance').fetchone()[0]
    conn.close()
    total_pages = (total_count + per_page - 1) // per_page
    return render_template('attendance.html', records=records, page=page, total_pages=total_pages)

@app.route('/delete_attendance/<int:id>')
def delete_attendance(id):
    if 'admin' not in session:
        return redirect('/')
    conn = get_db_connection()
    conn.execute('DELETE FROM attendance WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash(' Attendance record deleted.', 'info')
    return redirect(url_for('view_attendance'))

@app.route('/fraud')
def view_fraud():
    if 'admin' not in session:
        return redirect('/')
    page = int(request.args.get('page', 1))
    per_page = 10
    offset = (page - 1) * per_page
    conn = get_db_connection()
    records = conn.execute('''
        SELECT * FROM fraud
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    ''', (per_page, offset)).fetchall()
    total_count = conn.execute('SELECT COUNT(*) FROM fraud').fetchone()[0]
    conn.close()
    total_pages = (total_count + per_page - 1) // per_page
    return render_template('fraud.html', records=records, page=page, total_pages=total_pages)

@app.route('/delete_fraud/<int:id>')
def delete_fraud(id):
    if 'admin' not in session:
        return redirect('/')
    conn = get_db_connection()
    image_row = conn.execute("SELECT image_path FROM fraud WHERE id = ?", (id,)).fetchone()
    if image_row and image_row['image_path'] and os.path.exists(image_row['image_path']):
        try:
            os.remove(image_row['image_path'])
        except Exception as e:
            print(f" Error deleting image: {e}")
    conn.execute('DELETE FROM fraud WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash(' Fraud record deleted.', 'info')
    return redirect(url_for('view_fraud'))

@app.route('/face_ticket')
def face_ticket():
    return render_template('face_ticket.html')

@app.route('/fraud/<path:filename>')
def serve_fraud_image(filename):
    return send_from_directory('fraud', filename)

# Add these globally
last_label = None
repeat_count = 0

@app.route('/process_face', methods=['GET'])
def process_face():
    from math import sqrt

    global last_label, repeat_count, fraud_delay_active, last_fraud_time, last_processed_time,SMS_ALERTS_ENABLED

    now_ist = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

    if time.time() - last_processed_time < 8:
        print("⏸ Cooldown active. Skipping detection.")
        return jsonify({'status': 'waiting'})

    if fraud_delay_active and time.time() - last_fraud_time < FRAUD_DELAY_SECONDS:
        return jsonify({'status': 'waiting'})
    elif fraud_delay_active:
        fraud_delay_active = False
        last_label = None
        repeat_count = 0

    # - Capture directly from camera
    success, color_image = camera.read()
    if not success:
        return jsonify({'status': 'error', 'message': 'Camera read failed'})

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    face_box = detect_single(gray)

    if face_box is None:
        print("- No face detected.")
        last_label = None
        repeat_count = 0
        return jsonify({'status': 'waiting'})

    label, score, annotated_frame = recognize_face_from_frame(color_image)
    print(f" Detected: {label} | Score: {score:.2f}")

    if label == last_label:
        repeat_count += 1
    else:
        repeat_count = 1
        last_label = label

    if repeat_count >= 3 and label != "Unknown":
        print("- Face recognized. Running LIVENESS test via eye closure...")
        publish_mqtt("recognized", "Face Valid. Close Your eyes for 3 Seconds for two factor Authentication")
        time.sleep(4)

        # LIVENESS CHECK
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        mp_drawing = mp.solutions.drawing_utils

        start_time = time.time()
        eyes_closed_start = None
        eyes_closed_total = 0

        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        EYE_AR_THRESHOLD = 0.25
        TOTAL_DURATION = 6
        CLOSE_REQUIRED = 3

        def calculate_EAR(landmarks, eye_indices, iw, ih):
            coords = [(int(landmarks[i].x * iw), int(landmarks[i].y * ih)) for i in eye_indices]
            v1 = sqrt((coords[1][0] - coords[5][0])**2 + (coords[1][1] - coords[5][1])**2)
            v2 = sqrt((coords[2][0] - coords[4][0])**2 + (coords[2][1] - coords[4][1])**2)
            h = sqrt((coords[0][0] - coords[3][0])**2 + (coords[0][1] - coords[3][1])**2)
            return (v1 + v2) / (2.0 * h)

        while time.time() - start_time < TOTAL_DURATION:
            success, live_frame = camera.read()
            if not success:
                break

            ih, iw = live_frame.shape[:2]
            rgb_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                #  Draw landmarks on the face
                for face_landmarks in result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=live_frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                # Proceed with EAR calculation
                landmarks = result.multi_face_landmarks[0].landmark
                left_ear = calculate_EAR(landmarks, LEFT_EYE, iw, ih)
                right_ear = calculate_EAR(landmarks, RIGHT_EYE, iw, ih)
                ear = (left_ear + right_ear) / 2.0

                print(f"[DEBUG] EAR: {ear:.3f}")

                if ear < EYE_AR_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                else:
                    if eyes_closed_start:
                        eyes_closed_total += time.time() - eyes_closed_start
                        eyes_closed_start = None

            # (Optional) If you want, you could send live_frame to frontend
            # e.g., via cv2.imencode and Flask streaming

            time.sleep(0.2)

        if eyes_closed_start:
            eyes_closed_total += time.time() - eyes_closed_start
            print(f"[DEBUG] Total Eyes Closed Time: {eyes_closed_total:.2f}s")

        publish_mqtt("recognized", " ")

        if eyes_closed_total >= CLOSE_REQUIRED:
            print("- LIVE FACE DETECTED")
            usn = label.split("_")[-1]
            conn = get_db_connection()
            student = conn.execute("SELECT * FROM students WHERE usn = ?", (usn,)).fetchone()
            if student:
                conn.execute("INSERT INTO attendance (usn, timestamp) VALUES (?, ?)", (usn, now_ist))
                conn.commit()
                conn.close()
                cv2.imwrite("last_detected.jpg", annotated_frame)
                last_label = None
                repeat_count = 0
                last_processed_time = time.time()
                return jsonify({
                    'status': 'recognized',
                    'name': student['name'],
                    'usn': student['usn'],
                    'hallticket': student['hallticket'],
                    'block': student['block'],
                    'message': 'Valid student. Proceed to Block ' + student['block']
                })
        else:
            print("- FAKE FACE DETECTED")
            fraud_path = f"fraud/fraud_{int(time.time())}.jpg"
            os.makedirs("fraud", exist_ok=True)
            cv2.imwrite(fraud_path, color_image)
            conn = get_db_connection()
            conn.execute("INSERT INTO fraud (image_path, timestamp) VALUES (?, ?)", (fraud_path, now_ist))
            conn.commit()
            conn.close()
            send_fraud_email(fraud_path)
            if(SMS_ALERTS_ENABLED==True):
                publish_mqttsms("+917795958917")
            fraud_delay_active = True
            last_fraud_time = time.time()
            last_processed_time = time.time()
            last_label = None
            repeat_count = 0
            return jsonify({
                'status': 'unauthorized',
                'message': 'Fake face.  Please contact admin.'
            })

    elif repeat_count >= 3 and label == "Unknown":
        print("Unauthorized face detected 3 times.")
        fraud_path = f"fraud/fraud_{int(time.time())}.jpg"
        os.makedirs("fraud", exist_ok=True)
        cv2.imwrite(fraud_path, color_image)
        conn = get_db_connection()
        conn.execute("INSERT INTO fraud (image_path, timestamp) VALUES (?, ?)", (fraud_path, now_ist))
        conn.commit()
        conn.close()
        send_fraud_email(fraud_path)
        if(SMS_ALERTS_ENABLED==True):
                publish_mqttsms("+917795958917")
        fraud_delay_active = True
        last_fraud_time = time.time()
        last_processed_time = time.time()
        last_label = None
        repeat_count = 0
        return jsonify({
            'status': 'unauthorized',
            'message': 'Unauthorized. Please contact admin.'
        })

    return jsonify({'status': 'waiting'})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

