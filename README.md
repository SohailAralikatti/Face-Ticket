# Face-Ticket  
## A Facial Recognition Approach for Exam Hall Ticket Management System

Face-Ticket is an **AI-based exam hall authentication system** designed to prevent impersonation and cheating during examinations. The system replaces traditional manual hall ticket and ID verification with **facial recognition and liveness detection**, ensuring secure, fast, and automated student verification.

---

## 📌 Problem Statement
Conventional exam hall verification methods rely on physical hall tickets and ID cards, which are vulnerable to:
- Impersonation and identity fraud  
- Manual verification errors  
- Time delays during exam entry  
- Increased workload for invigilators  

Face-Ticket addresses these challenges using **biometric facial authentication** powered by artificial intelligence.

---

## 🚀 Solution Overview
The Face-Ticket system captures a student’s facial image in real time at the exam hall entry point and verifies it against pre-registered facial data using **CNN-based deep learning models**.  

Once authenticated:
- The student’s exam hall details are displayed  
- Attendance is automatically marked  
- Unauthorized attempts are detected and logged  

To prevent spoofing using photos or videos, the system integrates **liveness detection using eye-blink analysis**.

---

## ✨ Key Features
- Facial recognition-based exam authentication  
- Automatic exam hall ticket display  
- Real-time attendance marking  
- Liveness detection to prevent spoofing  
- Fraud detection with image capture  
- Email alerts for impersonation attempts  
- Admin dashboard for student and attendance management  
- Secure storage of biometric and exam data  

---

## 🛠️ Technologies Used

### 🔹 Programming Language
- Python

### 🔹 Web Framework
- Flask

### 🔹 Frontend Technologies
- HTML  
- CSS  
- Jinja2 Templates  

### 🔹 Machine Learning & Computer Vision
- TensorFlow / Keras (CNN-based face recognition)
- OpenCV (face detection and image processing)
- MediaPipe FaceMesh (liveness detection)

### 🔹 Database
- SQLite / MySQL

### 🔹 Development Tools
- Visual Studio Code / PyCharm  
- Anaconda Navigator  
- Web Browser (Chrome / Firefox)

---

## 🖥️ Platform Compatibility
⚠️ **This project is supported and tested only on Linux-based operating systems.**

Due to dependencies related to:
- OpenCV webcam access  
- MediaPipe FaceMesh  
- Real-time video processing  

the Face-Ticket system runs reliably only on Linux distributions such as:
- Ubuntu  
- Debian  
- Linux Mint  

Running this project on Windows or macOS may result in camera, driver, or library compatibility issues.

---

## 🧠 System Workflow
1. Student approaches the exam hall kiosk  
2. Webcam captures the facial image  
3. Face is verified using CNN-based recognition  
4. Liveness detection confirms a real person  
5. Attendance is marked automatically  
6. Exam hall details are displayed  
7. Fraud attempts trigger alerts and logs  

---

## 📂 Project Structure

Face-Ticket
│── app.py
│── templates/
│── static/
│── dataset/
│── models/
│── database/
│── fraud/
│── .gitignore
│── README.md


---

## ▶️ How to Run the Project (Linux Only)

1. Clone the repository:
   ```bash
   git clone https://github.com/SohailAralikatti/Face-Ticket.git


2. Navigate to the project directory:

cd Face-Ticket


3. Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate


4. Install required dependencies:

pip install -r requirements.txt


5. Run the Flask application:

python app.py


6. Open a browser and visit:

http://127.0.0.1:5000
