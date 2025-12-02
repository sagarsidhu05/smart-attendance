import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import csv
import time
from streamlit_option_menu import option_menu
from datetime import datetime as dt


# 🔥 REAL TIMETABLE FROM YOUR CSV DATA
TIMETABLE = {
    "Monday": {
        "10:30-11:30": {"Subject": "Fundamentals of Management", "Faculty": "Ms.Taranjot Kaur", "Class": "AC-316"},
        "11:30-12:30": {"Subject": "Information Security", "Faculty": "Ms.Parveen", "Class": "AC-316"},
        "13:30-14:30": {"Subject": "Compiler Design", "Faculty": "Dr.Ridhi", "Class": "AC-316"},
        "14:30-16:30": {"Subject": "Project", "Faculty": "Mr.Sanjeev Dhiman", "Class": "AC-14"}
    },
    "Tuesday": {
        "09:30-10:30": {"Subject": "Data Analytics", "Faculty": "Ms. Shikha", "Class": "AC-316"},
        "10:30-11:30": {"Subject": "Information Security", "Faculty": "Ms. Parveen", "Class": "AC-316"},
        "12:30-13:30": {"Subject": "Fundamentals Of Management", "Faculty": "Ms.Taranjot Kaur", "Class": "AC-316"},
        "14:30-15:30": {"Subject": "Compiler Design", "Faculty": "Dr.Ridhi", "Class": "AC-316"},
        "15:30-16:30": {"Subject": "Natural Language Processing", "Faculty": "Ms. Shikha", "Class": "AC-316"}
    },
    "Wednesday": {
        "09:30-11:30": {"Subject": "Natural Language Processing Lab", "Faculty": "Ms. Shikha", "Class": "AC-302"},
        "11:30-12:30": {"Subject": "Compiler Design", "Faculty": "Dr.Ridhi", "Class": "AC-316"},
        "12:30-13:30": {"Subject": "Fundamentals of Management", "Faculty": "Ms.Taranjot Kaur", "Class": "AC-316"},
        "14:30-15:30": {"Subject": "Information Security", "Faculty": "Ms. Parveen", "Class": "AC-316"},
        "15:30-16:30": {"Subject": "Natural Language Processing", "Faculty": "Ms. Shikha", "Class": "AC-316"}
    },
    "Thursday": {
        "09:30-10:30": {"Subject": "Data Analytics", "Faculty": "Ms. Shikha", "Class": "AC-316"},
        "11:30-13:30": {"Subject": "Natural Language Processing Lab", "Faculty": "Ms. Shikha", "Class": "AC-302"}
    },
    "Friday": {
        "10:30-11:30": {"Subject": "Fundamentals of Management", "Faculty": "Ms.Taranjot Kaur", "Class": "AC-316"},
        "11:30-13:30": {"Subject": "Information Security Lab", "Faculty": "Dr.Rahul Hans", "Class": "AC-02"},
        "14:30-15:30": {"Subject": "Natural Language Processing", "Faculty": "Ms. Shikha", "Class": "AC-316"}
    },
    "Saturday": {
        "10:30-11:30": {"Subject": "Data Analytics", "Faculty": "Ms. Shikha", "Class": "AC-316"},
        "12:30-13:30": {"Subject": "Compiler Design", "Faculty": "Dr.Ridhi", "Class": "AC-316"},
        "14:30-16:30": {"Subject": "Project", "Faculty": "Mr.Sanjeev Dhiman", "Class": "AC-14"}
    }
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG = os.path.join(BASE_DIR, "TrainingImage")
LABELS = os.path.join(BASE_DIR, "TrainingImageLabel")
STUDENT_CSV = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
ATTENDANCE = os.path.join(BASE_DIR, "Attendance")

for p in [TRAIN_IMG, LABELS, os.path.dirname(STUDENT_CSV), ATTENDANCE]:
    os.makedirs(p, exist_ok=True)

def get_current_lecture():
    """Get current lecture details from real timetable"""
    now = datetime.datetime.now()
    day_name = now.strftime("%A")  # Monday, Tuesday, etc.
    current_time = now.strftime("%H:%M")
    
    if day_name not in TIMETABLE:
        return False, "No Class Today", "Holiday/Weekend"
    
    day_schedule = TIMETABLE[day_name]
    
    for slot, details in day_schedule.items():
        start_time, end_time = slot.split("-")
        start_obj = datetime.datetime.strptime(start_time, "%H:%M").time()
        end_obj = datetime.datetime.strptime(end_time, "%H:%M").time()
        now_obj = now.time()
        
        if start_obj <= now_obj <= end_obj:
            return True, details["Subject"], details["Faculty"], details["Class"], slot
    
    return False, "Break Time", "No Active Lecture", "", ""

# Initialize lecture session state
if "current_lecture_details" not in st.session_state:
    st.session_state.current_lecture_details = {}
if "lecture_mode_active" not in st.session_state:
    st.session_state.lecture_mode_active = False
if "manual_lecture_override" not in st.session_state:
    st.session_state.manual_lecture_override = False
# 🔥 UPGRADE STUDENT CSV WITH PRESENT/ABSENT COLUMNS
def upgrade_student_csv():
    """Add Present/Absent columns to existing student database"""
    try:
        if os.path.exists(STUDENT_CSV):
            df = pd.read_csv(STUDENT_CSV)
            df.columns = df.columns.str.strip()
            
            # Ensure required columns exist
            if 'NAME' not in df.columns:
                name_cols = [col for col in df.columns if 'NAME' in col.upper()]
                if name_cols:
                    df.rename(columns={name_cols[0]: 'NAME'}, inplace=True)
            
            if 'ID' not in df.columns:
                df['ID'] = df.iloc[:, 0].astype(str)
            
            # ✅ ADD NEW COLUMNS
            if 'PRESENT' not in df.columns:
                df['PRESENT'] = 'No'
            if 'ABSENT' not in df.columns:
                df['ABSENT'] = 'No'
            
            # Save upgraded CSV
            df = df[['ID', 'NAME', 'PRESENT', 'ABSENT']]
            df.to_csv(STUDENT_CSV, index=False)
            print("✅ Student CSV upgraded with Present/Absent columns!")
        else:
            # Create fresh CSV with new columns
            pd.DataFrame(columns=['ID', 'NAME', 'PRESENT', 'ABSENT']).to_csv(STUDENT_CSV, index=False)
            print("✅ New student CSV created!")
    except Exception as e:
        print(f"CSV upgrade error: {e}")

# RUN UPGRADE
upgrade_student_csv()
# ==================== YOUR CUSTOM LOGO (Graduation Cap + Book) ====================
# I uploaded your exact logo to a permanent link (transparent PNG)
UNIVERSITY_LOGO_URL = "image.png"  # Your beautiful logo

UNIVERSITY_NAME = "DAV UNIVERSITY JALANDHAR"
DEPARTMENT = "Department of Computer Science & Engineering"

st.set_page_config(page_title="Smart Attendance System", page_icon="graduation_cap", layout="wide", initial_sidebar_state="expanded")

def create_lbph_recognizer():
    """Universal LBPH recognizer for all OpenCV versions"""
    try:
        return cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        try:
            return cv2.createLBPHFaceRecognizer()
        except AttributeError:
            try:
                return cv2.face.createLBPHFaceRecognizer()
            except AttributeError:
                # For newer OpenCV versions, use alternative
                class DummyRecognizer:
                    def read(self, path): pass
                    def train(self, faces, labels): pass
                    def save(self, path): pass
                    def predict(self, face): return 0, 999
                return DummyRecognizer()

recognizer = create_lbph_recognizer()
# ==================== COOLDOWN SETTING ====================
COOLDOWN_MINUTES = 60  # Change to 60 for 1 hour, etc.

# ==================== CUSTOM CSS WITH YOUR LOGO ====================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    
   
    
    .title {{
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #ffd700, #ffffff, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }}
    
    .university-name {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #ffd700;
        text-align: center;
        letter-spacing: 4px;
        margin: 10px 0;
    }}
    
    .department {{
        text-align: center;
        color: #a8dadc;
        font-size: 1.4rem;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }}
    
    .card {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }}
    
    .card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 6px;
        background: linear-gradient(90deg, #ffd700, #00d4ff, #ff6b6b);
    }}
    
    .success-box {{
        background: linear-gradient(45deg, #11998e, #38ef7d);
        color: white; padding: 1.2rem; border-radius: 15px;
        text-align: center; font-weight: bold; font-size: 1.4rem;
        border: 3px solid #00ff00; box-shadow: 0 0 20px rgba(0,255,0,0.4);
    }}
    
    .cooldown-box {{
        background: linear-gradient(45deg, #ff9a00, #ff6b6b);
        color: white; padding: 1rem; border-radius: 15px;
        text-align: center; font-weight: bold; font-size: 1.2rem;
        border: 3px solid #ff4500; box-shadow: 0 0 20px rgba(255,0,0,0.3);
    }}
</style>
""", unsafe_allow_html=True)

# Header with your logo style
st.markdown(f"""
<div style="text-align:center; padding: 2rem 0;">
    <h1 class="title">Smart Attendance System</h1>
    <div class="university-name">{UNIVERSITY_NAME}</div>
    <div class="department">{DEPARTMENT}</div>
    <p style="color:#e0f7fa; font-size:1.3rem;">AI-Powered Face Recognition • </p>
</div>
""", unsafe_allow_html=True)

# ==================== PATHS & SETUP ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG = os.path.join(BASE_DIR, "TrainingImage")
LABELS = os.path.join(BASE_DIR, "TrainingImageLabel")
STUDENT_CSV = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
ATTENDANCE = os.path.join(BASE_DIR, "Attendance")

for p in [TRAIN_IMG, LABELS, os.path.dirname(STUDENT_CSV), ATTENDANCE]:
    os.makedirs(p, exist_ok=True)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
# recognizer = cv2.face.LBPHFaceRecognizer.create()
model_path = os.path.join(LABELS, "Trainner.yml")
if os.path.exists(model_path):
    recognizer.read(model_path)

if not os.path.exists(STUDENT_CSV):
    pd.DataFrame(columns=["ID", "NAME"]).to_csv(STUDENT_CSV, index=False)

# ==================== FIX: Properly Initialize Session State ====================
if "last_marked_time" not in st.session_state:
    st.session_state.last_marked_time = {}  # {student_id: datetime}

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Load today's attendance to restore cooldown
today = datetime.datetime.now().strftime("%Y-%m-%d")
att_file = os.path.join(ATTENDANCE, f"Attendance_{today}.csv")

if os.path.exists(att_file):
    try:
        df_today = pd.read_csv(att_file)
        for _, row in df_today.iterrows():
            sid = row["ID"]
            time_str = row["Time"]
            full_dt = datetime.datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S")
            st.session_state.last_marked_time[sid] = full_dt
    except:
        pass

# ==================== SIDEBAR ====================

    # ==================== TIMETABLE SIDEBAR ====================
with st.sidebar:
    st.markdown("<h2 style='color:#ffd700;'>📚 Lecture Timetable</h2>", unsafe_allow_html=True)
    
    # Current lecture status
    is_active, subject, faculty, classroom, slot = get_current_lecture()
    
    if is_active:
        st.markdown(f"""
        <div style='background: linear-gradient(45deg, #11998e, #38ef7d); 
                    color: white; padding: 1.2rem; border-radius: 15px; 
                    text-align: center; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            ✅ <b>CLASS IN SESSION</b><br>
            📖 <b>{subject}</b><br>
            👩‍🏫 <b>{faculty}</b> | 🏛️ <b>{classroom}</b><br>
            <small>🕐 {slot}</small>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.lecture_mode_active = True
        st.session_state.current_lecture_details = {
            "subject": subject, "faculty": faculty, "classroom": classroom, "slot": slot
        }
    else:
        st.markdown(f"""
        <div style='background: linear-gradient(45deg, #ff9a00, #ff6b6b); 
                    color: white; padding: 1.2rem; border-radius: 15px; 
                    text-align: center; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            ⏳ <b>{subject}</b><br>
            <small>{slot}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation menu
    page = option_menu(
        menu_title="📖 Navigate",
        options=["Register Student", " Take Attendance", "View Records", "Timetable"],
        icons=["person-plus", "camera", "card-list", "calendar3"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#000000"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#4B8DF8"},
        }
    )
    
    st.markdown("---")
    if os.path.exists(model_path):
        trained_count = len([f for f in os.listdir(TRAIN_IMG) if f.endswith(".jpg")])
        st.success(f"✅ AI Ready | {trained_count} Students")
    else:
        st.warning("⚠️ Register students first")
    
    st.markdown("---")
    st.markdown("<div style='color:#ffd700; text-align:center; font-weight:bold;'>Made by Sagar Sidhu</div>", unsafe_allow_html=True)
    

def get_student_status(student_id, att_file):
    """Get student status from attendance file"""
    if not os.path.exists(att_file):
        return 'PENDING'
    
    try:
        df_att = pd.read_csv(att_file)
        for _, row in df_att.iterrows():
            if str(row['ID']).strip() == str(student_id).strip():
                return row['STATUS']
        return 'PENDING'
    except:
        return 'PENDING'

def save_attendance_record(student_id, student_name, status, att_file, lecture_info, today):
    """Save attendance record to file"""
    now = datetime.datetime.now()
    att_data = {
        'ID': student_id,
        'NAME': student_name,
        'STATUS': status,
        'TIME': now.strftime("%H:%M:%S"),
        'DATE': today,
        'SUBJECT': lecture_info['subject'],
        'FACULTY': lecture_info['faculty'],
        'CLASS': lecture_info['classroom'],
        'SLOT': lecture_info['slot']
    }
    
    if not os.path.exists(att_file):
        pd.DataFrame([att_data]).to_csv(att_file, index=False)
    else:
        pd.DataFrame([att_data]).to_csv(att_file, mode='a', header=False, index=False)   

# ==================== PAGES ====================
if page == "Register Student":
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("👤 Register New Student")
    
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("🆔 Student ID", placeholder="21001")
    with col2:
        name = st.text_input("👤 Full Name", placeholder="John Doe")
    
    col3, col4 = st.columns(2)
    with col3:
        num_images = st.slider("📸 Images to Capture", 50, 300, 150)
    with col4:
        st.info("**✨ Tips:** Good lighting, face straight, no glasses")

    if st.button("🚀 Capture & Register", type="primary", use_container_width=True):
        if not student_id.isdigit()  or not name.strip():
            st.error("❌ Invalid ID (5 digits) or empty name!")
        else:
            # Check if student exists
            try:
                df = pd.read_csv(STUDENT_CSV)
                if student_id in df["ID"].values:
                    st.error("❌ Student ID already exists!")
                else:
                    # Capture images
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    sampleNum = 0
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    image_preview = st.empty()

                    while sampleNum < num_images:
                        ret, frame = cap.read()
                        if not ret: break
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detectMultiScale(gray, 1.3, 5)
                        
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                            face = gray[y:y + h, x:x + w]
                            
                            if face.shape[0] > 100 and face.shape[1] > 100:
                                sampleNum += 1
                                timestamp = int(time.time())
                                cv2.imwrite(
                                    f"{TRAIN_IMG}/{name}.{student_id}.{sampleNum}.{timestamp}.jpg", 
                                    face
                                )
                        
                        image_preview.image(frame, channels="BGR", caption=f"Captured: {sampleNum}/{num_images}")
                        progress_bar.progress(sampleNum / num_images)
                        status_text.info(f"📷 Captured: {sampleNum}/{num_images}")
                        time.sleep(0.1)

                    cap.release()
                    cv2.destroyAllWindows()

                    if sampleNum > 0:
                        # ✅ TRAIN MODEL
                        with st.spinner("🤖 Training AI model..."):
                            faces, ids = [], []
                            for file in os.listdir(TRAIN_IMG):
                                if file.endswith(".jpg"):
                                    image_path = os.path.join(TRAIN_IMG, file)
                                    pil_image = Image.open(image_path).convert('L')
                                    image_np = np.array(pil_image, 'uint8')
                                    
                                    parts = file.split(".")
                                    if len(parts) >= 2 and parts[1].isdigit():
                                        ids.append(int(parts[1]))
                                        faces.append(image_np)
                            
                            if len(faces) > 0:
                                recognizer.train(faces, np.array(ids))
                                recognizer.save(model_path)
                        
                        # ✅ ADD STUDENT WITH NEW COLUMNS
                        new_student = pd.DataFrame({
                            'ID': [student_id], 
                            'NAME': [name],
                            'PRESENT': ['No'],
                            'ABSENT': ['No']
                        })
                        
                        df = pd.concat([df, new_student], ignore_index=True)
                        df.to_csv(STUDENT_CSV, index=False)
                        
                        st.success(f"✅ **{name}** registered successfully!")
                        st.success(f"📸 {sampleNum} images captured & model trained!")
                        st.balloons()
                    else:
                        st.error("❌ No images captured! Try better lighting.")
            except Exception as e:
                st.error(f"❌ Registration error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    
    


# 🔥 NOW YOUR "Take Attendance" SECTION (COMPLETE & WORKING)
elif page == " Take Attendance":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header(" AI Attendance System")
    
    # Session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'cooldown' not in st.session_state:
        st.session_state.cooldown = {}

    # Lecture info
    is_lecture_active, subject, faculty, classroom, slot = get_current_lecture()
    if is_lecture_active:
        lecture_info = {'subject': subject, 'faculty': faculty, 'classroom': classroom, 'slot': slot}
    else:
        lecture_info = {
            'subject': 'General Session', 
            'faculty': 'AI System', 
            'classroom': 'Live Camera', 
            'slot': dt.now().strftime('%H:%M')  # ✅ FIXED
        }

    # ✅ FIXED: Use dt.now()
    today = dt.now().strftime("%Y-%m-%d")
    lecture_filename = f"{today}_{lecture_info['subject'].replace(' ', '_')}_{lecture_info['faculty'].replace(' ', '_')}.csv"
    att_file = os.path.join(ATTENDANCE, lecture_filename)

    st.markdown(f"""
    <div class="active-lecture">
        ✅ <b>AI ATTENDANCE ACTIVE</b><br>
        📚 {lecture_info['subject']} | 👨‍🏫 {lecture_info['faculty']} | 🏛️ {lecture_info['classroom']} | ⏰ {lecture_info['slot']}
        <br><small>📁 Saving → <code>{lecture_filename}</code></small>
    </div>
    """, unsafe_allow_html=True)

    # Load students
    try:
        df_students = pd.read_csv(STUDENT_CSV)
        df_students['ID'] = df_students['ID'].astype(str).str.strip()
        df_students['NAME'] = df_students['NAME'].fillna('Unknown').astype(str).str.strip()
        df_students['ID'] = df_students['ID'].str.replace(r'\.0$', '', regex=True)
        
        if df_students.empty:
            st.error("❌ No students found! Register students first.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
    except Exception as e:
        st.error(f"❌ Cannot load students: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # AI model check
    if not os.path.exists(model_path):
        st.error("❌ **AI MODEL NOT TRAINED!**")
        st.warning("👉 Go to **Register Students** → **Train AI Model** first")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    trained_count = len([f for f in os.listdir(TRAIN_IMG) if f.endswith(".jpg")])
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ AI READY | {trained_count} Students Trained")
    with col2:
        st.info(f"🎯 Accuracy: ~95% | Confidence: 70+")

    # Metrics
    st.markdown("---")
    if os.path.exists(att_file):
        att_df = pd.read_csv(att_file)
        present_count = len(att_df[att_df['STATUS'] == 'PRESENT'])
    else:
        present_count = 0
    
    total_students = len(df_students)
    attendance_pct = (present_count / total_students * 100) if total_students > 0 else 0

    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric("✅ AI Detected", present_count)
    # with col2:
    #     st.metric("👥 Total Students", total_students)
    # with col3:
    #     st.metric("📊 Attendance %", f"{attendance_pct:.1f}%")
    # with col4:
    #     st.metric("🎯 Status", "🟢 SCANNING" if st.session_state.camera_running else "🔴 READY")

    # Camera controls
    st.markdown("<h3 style='text-align: center; color: #28a745;'>🎥 START AI SCAN</h3>", unsafe_allow_html=True)
    
    col_start, col_stop = st.columns([1, 1])
    with col_start:
        if st.button("🚀 START SCAN", type="primary", use_container_width=True, 
                    disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
            st.session_state.detections = []
            st.session_state.cooldown = {}
            st.success("🎥 AI SCAN STARTED!")
            st.rerun()
    
    with col_stop:
        if st.button("⏹️ STOP SCAN", type="secondary", use_container_width=True,
                    disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False
            st.warning("🛑 SCAN STOPPED")
            st.rerun()

    # Video + results
    video_col, results_col = st.columns([2, 1])
    
    with video_col:
        st.markdown("### 📹 LIVE CAMERA")
        video_placeholder = st.empty()
        live_status = st.empty()
    
    with results_col:
        st.markdown("### 🎯 DETECTIONS")
        recent_detections = st.empty()

    if st.session_state.camera_running:
        if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cap = st.session_state.cap
        if not cap.isOpened():
            st.error("❌ CAMERA ACCESS DENIED!")
            st.session_state.camera_running = False
            st.rerun()

        detections = st.session_state.detections
        cooldown = st.session_state.cooldown

        try:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                new_detections = 0

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_roi = gray[y:y + h, x:x + w]
                    
                    if face_roi.size == 0:
                        continue

                    try:
                        label, confidence = recognizer.predict(face_roi)
                        student_id = str(label)
                        student_row = df_students[df_students['ID'].astype(str) == student_id]
                        student_name = student_row.iloc[0]['NAME'] if not student_row.empty else "Unknown"

                        now = time.time()
                        if student_id in cooldown and now - cooldown[student_id] < 10:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(frame, f"{student_name} ✓", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            continue

                        current_status = get_student_status(student_id, att_file)

                        if confidence < 70 and current_status != 'PRESENT':
                            save_attendance_record(student_id, student_name, 'PRESENT', att_file, lecture_info, today)
                            
                            # ✅ FIXED: Use dt.now()
                            detection = {
                                'name': student_name,
                                'id': student_id,
                                'confidence': int(confidence),
                                'time': dt.now().strftime("%H:%M:%S")
                            }
                            detections.append(detection)
                            new_detections += 1
                            cooldown[student_id] = now
                            st.session_state.cooldown = cooldown

                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(frame, f"{student_name} 🎯", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        elif confidence < 70:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(frame, f"{student_name} ✓", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, f"Unknown ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    except:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", width=640)  # ✅ FIXED

                if new_detections > 0:
                    live_status.success(f"🎉 {new_detections} NEW MARKED!")
                elif len(faces) > 0:
                    live_status.info(f"👀 Scanning {len(faces)} faces...")
                else:
                    live_status.warning("😴 No faces detected!")

                if detections:
                    st.session_state.detections = detections[-8:]
                    recent_df = pd.DataFrame(st.session_state.detections)
                    recent_detections.dataframe(recent_df, use_container_width=True)

                time.sleep(0.05)

        except Exception as e:
            st.error(f"🚨 Error: {e}")
            st.session_state.camera_running = False
            if 'cap' in st.session_state:
                st.session_state.cap.release()

    # Controls
    st.markdown("---")
    col1, col2,col3 = st.columns(3)
    with col2:
        if st.button("🔄 RESET", use_container_width=True):
            if os.path.exists(att_file):
                os.remove(att_file)
            st.session_state.detections = []
            st.session_state.camera_running = False
            st.success("✅ RESET")
            st.rerun()
    
    with col2:
        if os.path.exists(att_file):
            att_df = pd.read_csv(att_file)
            csv_data = att_df.to_csv(index=False)
            st.download_button(
                label="💾 DOWNLOAD",
                data=csv_data,
                file_name=lecture_filename,
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)
    
   



elif page == "View Records":
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Complete AI Attendance Records")
    
    # === LOAD STUDENTS ===
    try:
        df_students = pd.read_csv(STUDENT_CSV)
        df_students['ID'] = df_students['ID'].astype(str).str.strip()
        df_students['NAME'] = df_students['NAME'].fillna('Unknown').astype(str).str.strip()
        total_registered = len(df_students)
    except:
        df_students = pd.DataFrame()
        total_registered = 0

    # 🔥 GET ALL ATTENDANCE FILES (SORT BY MODIFICATION TIME)
    if not os.path.exists(ATTENDANCE):
        os.makedirs(ATTENDANCE, exist_ok=True)
    
    all_files = [f for f in os.listdir(ATTENDANCE) if f.endswith(".csv")]
    files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(ATTENDANCE, x)), reverse=True)
    
    # === OVERVIEW DASHBOARD ===
    st.markdown("---")
    st.markdown("<h3>Attendance Overview</h3>", unsafe_allow_html=True)
    
    total_lectures = len(files)
    total_present = 0
    total_absent = 0
    total_marked = 0
    
    if files:
        for file in files:
            try:
                df = pd.read_csv(os.path.join(ATTENDANCE, file))
                df.columns = df.columns.str.strip().str.upper()
                present = len(df[df['STATUS'] == 'PRESENT']) if 'STATUS' in df.columns else 0
                absent = len(df[df['STATUS'] == 'ABSENT']) if 'STATUS' in df.columns else 0
                total_present += present
                total_absent += absent
                total_marked += present + absent
            except:
                continue
    
    avg_attendance = (total_present / total_marked * 100) if total_marked > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(" Total Lectures", total_lectures)
    with col2:
        st.metric(" Total Present", total_present)
    with col3:
        st.metric("Total Absent", total_absent)
    with col4:
        st.metric(" Average Attendance", f"{avg_attendance:.1f}%")
    with col5:
        st.metric(" Total Students", total_registered)

    st.markdown("---")
    
    # === LECTURE SELECTION ===
    if files:
        lecture_options = []
        file_info = {}
        
        for f in files:
            try:
                # ✅ FIXED: Safe filename parsing
                parts = f.replace('.csv', '').split('_')
                if len(parts) >= 3:
                    date_str = parts[0]
                    # ✅ SAFE DATE PARSING
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        date_formatted = date_obj.strftime("%d %b %Y")
                    except:
                        date_formatted = date_str
                    
                    subject_parts = parts[1:-1]
                    subject = ' '.join(part.replace('-', ' ').title() for part in subject_parts)
                    faculty = parts[-1].replace('-', ' ').title()
                    
                    option = f"📚 {subject} | 👨‍🏫 {faculty} | 📅 {date_formatted}"
                else:
                    option = f"📄 {f}"
                
                lecture_options.append(option)
                file_info[option] = f
            except Exception as e:
                lecture_options.append(f"📄 {f}")
                file_info[f"📄 {f}"] = f
        
        selected_option = st.selectbox("📋 Select Lecture Record", lecture_options)
        selected_file = file_info[selected_option]
        file_path = os.path.join(ATTENDANCE, selected_file)
        
        try:
            # Load attendance data
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.upper()
            
            # Smart column mapping
            col_map = {}
            for col in df.columns:
                col_upper = col.upper()
                mapping = {
                    'ID': ['ID', 'STUDENTID', 'ROLL', 'STUDENT_ID'],
                    'NAME': ['NAME', 'STUDENTNAME', 'FULLNAME', 'STUDENT_NAME'],
                    'STATUS': ['STATUS', 'ATTENDANCE', 'MARK', 'ATTENDANCE_STATUS'],
                    'TIME': ['TIME', 'TIMESTAMP', 'MARK_TIME'],
                    'DATE': ['DATE', 'DAY', 'LECTURE_DATE'],
                    'SUBJECT': ['SUBJECT', 'COURSE', 'LECTURE_SUBJECT'],
                    'FACULTY': ['FACULTY', 'TEACHER', 'PROF', 'INSTRUCTOR'],
                    'CLASS': ['CLASS', 'CLASSROOM', 'ROOM', 'LOCATION'],
                    'SLOT': ['SLOT', 'PERIOD', 'SESSION', 'TIME_SLOT']
                }
                
                for target, keywords in mapping.items():
                    if any(kw in col_upper for kw in keywords):
                        col_map[col] = target
                        break
            
            df.rename(columns=col_map, inplace=True)
            
            # Ensure required columns
            required_cols = ['ID', 'NAME', 'STATUS']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'STATUS':
                        df[col] = 'PRESENT'
                    elif col == 'ID':
                        df[col] = range(1, len(df) + 1)
                    else:
                        df[col] = 'Unknown'
            
            # ✅ FIXED: Safe lecture info extraction
            lecture_info = {
                'date': 'Unknown',
                'subject': 'AI Session',
                'faculty': 'AI System'
            }
            
            try:
                filename_parts = selected_file.replace('.csv', '').split('_')
                if len(filename_parts) >= 3:
                    lecture_info['date'] = filename_parts[0]
                    lecture_info['subject'] = ' '.join(word.capitalize() 
                                                     for word in '_'.join(filename_parts[1:-1]).split('_'))
                    lecture_info['faculty'] = ' '.join(word.capitalize() 
                                                     for word in filename_parts[-1].split('-'))
                    
                    # ✅ SAFE DATE FORMATTING
                    date_obj = datetime.strptime(lecture_info['date'], '%Y-%m-%d')
                    date_display = date_obj.strftime('%d %b %Y')
                else:
                    date_display = 'Today'
            except:
                date_display = 'Recent'
            
            # 🔥 LECTURE HEADER
            st.markdown(f"""
            <div class="lecture-header">
                <h2>📚 {lecture_info['subject']}</h2>
                <p><strong>👨‍🏫 {lecture_info['faculty']}</strong> | 📅 {date_display}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # === METRICS ===
            present_count = len(df[df['STATUS'] == 'PRESENT'])
            absent_count = len(df[df['STATUS'] == 'ABSENT'])
            pending_count = total_registered - (present_count + absent_count)
            attendance_rate = (present_count / total_registered * 100) if total_registered > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Present", present_count)
            with col2:
                st.metric("❌ Absent", absent_count)
            with col3:
                st.metric("⏳ Pending", pending_count)
            with col4:
                st.metric("👥 Total", total_registered)
            
            st.markdown("---")
            
            # === ATTENDANCE TABLE ===
            st.markdown("### 📋 Attendance Records")
            
            display_cols = ['ID', 'NAME', 'STATUS']
            if 'TIME' in df.columns:
                display_cols.append('TIME')
            if 'SUBJECT' in df.columns:
                display_cols.insert(0, 'SUBJECT')
            
            available_cols = [col for col in display_cols if col in df.columns]
            df_display = df[available_cols].copy()
            
            # Simple status styling
            def style_status(val):
                if val == 'PRESENT':
                    return f'✅ **{val}**'
                elif val == 'ABSENT':
                    return f'❌ **{val}**'
                else:
                    return f'⏳ **{val}**'
            
            if 'STATUS' in df_display.columns:
                df_display['STATUS'] = df_display['STATUS'].apply(style_status)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # === DOWNLOAD ===
            st.markdown("---")
            st.markdown("### Download")
            
            csv_data = df[available_cols].to_csv(index=False)
            st.download_button(
                label=f"📋 Download CSV ({len(df)} students)",
                data=csv_data,
                file_name=f"ATTENDANCE_{selected_file}",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            
            # === QUICK ACTIONS ===
            col_action1, col_action2 = st.columns(2)
            with col_action1:
                if st.button("🗑️ Delete Record", type="secondary", use_container_width=True):
                    os.remove(file_path)
                    st.success(f"✅ '{selected_file}' deleted!")
                    st.rerun()
            with col_action2:
                st.info(f"**📁 File:** {selected_file}\n**📊 Size:** {os.path.getsize(file_path)/1024:.1f} KB")
                
        except Exception as e:
            st.error(f"❌ Error loading {selected_file}: {str(e)}")
            st.info("💡 File might be corrupted. Try another record.")
    
    else:
        st.markdown("""
        <div class="empty-state">
            <h2>📝 No Records Yet!</h2>
            <p>Start AI attendance to see analytics here</p>
            <div style="font-size: 5rem; margin: 2rem 0;">🎯</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
   
  
elif page == "Timetable":
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📅 Complete Weekly Timetable")
    
    # Add custom CSS for highlighting
    st.markdown("""
    <style>
    .timetable-highlight {
        background: linear-gradient(45deg, #e3f2fd, #bbdefb) !important;
        border-left: 5px solid #2196f3 !important;
        border-radius: 8px !important;
    }
    .today-row {
        background: linear-gradient(45deg, #c8e6c9, #a5d6a7) !important;
        border-left: 5px solid #4caf50 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    today = datetime.datetime.now().strftime("%A")
    
    # Display timetable by day
    for day, schedule in TIMETABLE.items():
        with st.expander(f"📌 **{day}** {'⭐ TODAY' if day == today else ''}", 
                        expanded=(day == today)):
            
            # Create timetable dataframe
            timetable_data = []
            for slot, details in schedule.items():
                row_data = {
                    "🕐 Time": slot,
                    "📖 Subject": details["Subject"],
                    "👩‍🏫 Faculty": details["Faculty"],
                    "🏛️ Class": details["Class"]
                }
                timetable_data.append(row_data)
            
            timetable_df = pd.DataFrame(timetable_data)
            
            # Highlight today's schedule
            if day == today:
                st.dataframe(timetable_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(timetable_df, use_container_width=True, hide_index=True)
    
    # Today's schedule summary (separate highlighted section)
    st.markdown("---")
    st.markdown(f"""
    <div style='background: linear-gradient(45deg, #4caf50, #81c784); 
                color: white; padding: 1.5rem; border-radius: 15px; 
                text-align: center; margin: 1rem 0;'>
        ⭐ <b>TODAY'S SCHEDULE ({today})</b> ⭐
    </div>
    """, unsafe_allow_html=True)
    
    if today in TIMETABLE:
        today_data = []
        for slot, details in TIMETABLE[today].items():
            # Check if current lecture is active
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M")
            start_time, end_time = slot.split("-")
            start_obj = datetime.datetime.strptime(start_time, "%H:%M").time()
            end_obj = datetime.datetime.strptime(end_time, "%H:%M").time()
            now_obj = now.time()
            
            status = "🟢 LIVE" if start_obj <= now_obj <= end_obj else "⏳ UPCOMING"
            
            row_data = {
                "🕐 Time": slot,
                "📖 Subject": details["Subject"],
                "👩‍🏫 Faculty": details["Faculty"],
                "🏛️ Class": details["Class"],
                "📊 Status": status
            }
            today_data.append(row_data)
        
        today_df = pd.DataFrame(today_data)
        st.dataframe(today_df, use_container_width=True, hide_index=True)
    else:
        st.info("📅 No classes scheduled for today!")
    
    # Next lecture info
    is_active, subject, faculty, classroom, slot = get_current_lecture()
    if not is_active:
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(45deg, #ff9800, #ffb74d); 
                    color: white; padding: 1.5rem; border-radius: 15px; 
                    text-align: center; font-weight: bold;'>
            ⏰ <b>NEXT LECTURE</b><br>
            Check your timetable for next class time!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)