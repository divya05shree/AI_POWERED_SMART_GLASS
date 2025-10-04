import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from fdlite import FaceDetection, FaceDetectionModel
import sqlite3
import os
import io
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. GEMINI CLIENT SETUP (No changes) ---
load_dotenv()
try:
    client = genai.Client()
    GEMINI_MODEL = 'gemini-2.5-flash'
    print("[INFO] Gemini API Client Initialized.")
except Exception as e:
    print(f"[ERROR] Could not initialize Gemini Client. Check your API Key. Error: {e}")
    client = None

# --- 2. CORE HELPER FUNCTIONS (TensorFlow/DB) ---

# Database Setup (Runs once)
conn = sqlite3.connect("faces.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()


def load_pb_model(pb_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return graph


def preprocess_face(face_img, size=112):
    face_img = cv2.resize(face_img, (size, size))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = (face_img - 0.5) / 0.5
    return np.expand_dims(face_img, axis=0)


def load_faces_db():
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    db = {}
    for name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        db[name] = emb
    return db


def register_face_db(name, embedding):
    emb_blob = embedding.astype(np.float32).tobytes()
    c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, emb_blob))
    conn.commit()
    print(f"[INFO] Registered {name} in database.")


def recognize_face(embedding, threshold=0.8):
    min_dist, identity = float("inf"), "Unknown"
    for name, db_emb in face_db.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist:
            min_dist, identity = dist, name
    return identity if min_dist < threshold else "Unknown"


# Gemini Function (No changes)
def get_scene_analysis(frame, full_prompt):
    if client is None: return "Gemini API Error: Client not initialized."
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    contents = [full_prompt, pil_image]
    try:
        config = types.GenerateContentConfig(temperature=0.1,
                                             max_output_tokens=8192 if "LONG SCENE DESCRIPTION" in full_prompt.upper() else 800)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        if response.text is None or not response.text.strip():
            if response.candidates and response.candidates[
                0].finish_reason.name == "SAFETY": return "API Blocked: Content safety filter activated."
            if response.candidates and response.candidates[
                0].finish_reason.name != "STOP": return f"API Blocked: Reason {response.candidates[0].finish_reason.name}"
            return "API returned empty response (None/Empty String)."
        return response.text.strip()
    except Exception as e:
        return f"API Call Failed: {e}"


# --- TENSORFLOW AND DATABASE INITIALIZATION ---
pb_path = "MobileFaceNet_TF/arch/pretrained_model/MobileFaceNet_9925_9680.pb"
graph = load_pb_model(pb_path)
sess = tf.compat.v1.Session(graph=graph)
input_tensor = graph.get_tensor_by_name("input:0")
embedding_tensor = graph.get_tensor_by_name("embeddings:0")
face_db = load_faces_db()
detector = FaceDetection(model_type=FaceDetectionModel.SHORT)

# --- SCENE DESCRIPTION BASE PROMPTS ---
LONG_DESC_BASE = "LONG SCENE DESCRIPTION: Provide a detailed and comprehensive description of the entire scene, including all noticeable objects, the environment, lighting, and any context."
SHORT_DESC_BASE = "SHORT SCENE SUMMARY: Provide a very concise, 1-2 sentence summary of the main objects and the overall environment."
OCR_PROMPT = "Scan the image for any visible numbers , if its currency then how much money it is has to be recognized and should explain this is this much rupees notes are there and  , if any numbers then describe it (e.g., license plates, clock time, signs, etc.). Output ONLY the numbers you find, separated by a comma. If no numbers are found, output 'N/A'."

# --- MAIN VIDEO CAPTURE LOOP ---
cap = cv2.VideoCapture(0)
print("\nCamera ready.")
print(
    "Controls: 'r' to register face, 'p' for NAME lookup (console), 's' for SHORT description, 'l' for LONG description (console), 'o' for OCR, 'q' to quit.")

# Loop Control Variables
last_api_time = 0
API_COOLDOWN = 3
current_short_desc = "Press 's' for scene summary..."
current_ocr_result = "Press 'o' for OCR numbers..."

# --- FR State Management Variables ---
FR_LOGGING_INTERVAL = 5.0  # Log/Recognize once every 5 seconds
last_log_time = 0
current_frame_identities = []  # Stores identities from the last successful FR run
last_embedding = None
last_logged_identity_summary = ""  # Key for state change check (e.g., "Alice,Unknown")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # 1. Continuous Face Detection (Fast Process)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = detector(pil_image)

    # 2. FIXED-INTERVAL RECOGNITION AND LOGGING CHECK
    trigger_fr_and_log = current_time - last_log_time > FR_LOGGING_INTERVAL

    if trigger_fr_and_log:

        # Reset tracking data for this new recognition run
        new_run_identities = []

        if faces:
            # --- RUN EXPENSIVE RECOGNITION ---

            for face in faces:
                h, w, _ = frame.shape
                box = face.bbox
                x1, y1 = int(box.xmin * w), int(box.ymin * h)
                x2, y2 = int(box.xmax * w), int(box.ymax * h)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue

                # Perform deep learning recognition
                face_input = preprocess_face(face_crop)
                embedding = sess.run(embedding_tensor, feed_dict={input_tensor: face_input})[0]
                identity = recognize_face(embedding)

                # Store results
                new_run_identities.append(identity)
                last_embedding = embedding  # Store the last embedding for registration

            # --- CONSTRUCT CURRENT STATE SUMMARY (For comparison) ---
            unique_names = sorted(list(set(new_run_identities)))
            current_state_summary = ",".join(unique_names)

            # --- LOGGING DECISION: Only log if the identity has changed ---
            if current_state_summary != last_logged_identity_summary:

                # 1. LOG JSON DATA (ONLY NAMES)
                names_to_log = [name for name in unique_names if name != "Unknown"]
                if "Unknown" in unique_names:
                    names_to_log.append("Unknown")

                json_output = {"persons": names_to_log}
                print(f"\n[FACE LOG] {json.dumps(json_output)}")
                print(f"[FR INFO] Identity change detected: {current_state_summary}")

                # 2. UPDATE STATE
                last_logged_identity_summary = current_state_summary
                current_frame_identities = new_run_identities  # Update cached list for later requests

            # Reset timer regardless of logging, since recognition was performed
            last_log_time = current_time

        elif last_logged_identity_summary != "0-":
            # LOG when people leave (empty state)
            print("\n[FACE LOG] {\"persons\": []}")
            print("[FR INFO] No faces detected. Clearing state.")

            last_logged_identity_summary = "0-"
            current_frame_identities = []
            last_log_time = current_time  # Reset timer

    # 3. CONSTRUCT DYNAMIC PROMPT CONTEXT (Uses the state from the last successful log)
    # The current_frame_identities list holds the results from the last successful recognition interval
    if current_frame_identities:
        # We must use last_logged_identity_summary for context as it's the verified state
        names_for_context = last_logged_identity_summary.split(',')

        person_context = (
            f"Note: The image contains person(s) identified as {', '.join(names_for_context)}. "
            "Please integrate this information into your description, ensuring you still detail the environment and other objects."
        )
    else:
        person_context = "Note: Face recognition did not identify any person. Describe the scene as usual."

    # 4. Display Video Feed (Unchanged)
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)

    cv2.putText(frame, "Scene: " + current_short_desc, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, "OCR: " + current_ocr_result, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Face Recognition and Multimodal Analysis", frame)

    key = cv2.waitKey(1) & 0xFF

    # 5. Handle Keypresses
    if key == ord("q"):
        break
    elif key == ord("r"):
        name = input("Enter name: ")
        if 'last_embedding' in locals() and last_embedding is not None:
            register_face_db(name, last_embedding)
            face_db = load_faces_db()
        else:
            print("[WARNING] Registration failed: Face data not captured yet. Wait 5 seconds and try again.")

    # --- PURE IDENTIFICATION MODE ---
    elif key == ord("p"):
        if last_logged_identity_summary != "0-":
            names_only = last_logged_identity_summary.replace(',', ', ')
            print(f"\n[IDENTIFICATION RESULT] Person(s) in frame: {names_only}")
        else:
            print("\n[IDENTIFICATION RESULT] No face data available from the last scan.")


    # Check cooldown before triggering Gemini API
    elif current_time - last_api_time > API_COOLDOWN:

        if key == ord("s"):  # Short Description
            print("\n[API] Requesting Short Scene Description...")
            current_short_desc = "Processing..."
            cv2.imshow("Face Recognition and Multimodal Analysis", frame)
            cv2.waitKey(1)

            full_prompt = f"{person_context} {SHORT_DESC_BASE}"
            current_short_desc = get_scene_analysis(frame, full_prompt)
            print(f"[RESULT] Short Desc: {current_short_desc}")
            last_api_time = current_time

        elif key == ord("l"):  # Long Description
            print("\n[API] Requesting LONG Scene Description...")

            full_prompt = f"{person_context} {LONG_DESC_BASE}"
            long_desc = get_scene_analysis(frame, full_prompt)
            print("-" * 50)
            print(f"LONG SCENE DESCRIPTION:\n{long_desc}")
            print("-" * 50)
            last_api_time = current_time

        elif key == ord("o"):  # OCR for Numbers
            print("\n[API] Requesting OCR for Numbers...")
            current_ocr_result = "Processing..."
            cv2.imshow("Face Recognition and Multimodal Analysis", frame)
            cv2.waitKey(1)

            full_prompt = OCR_PROMPT
            current_ocr_result = get_scene_analysis(frame, full_prompt)
            print(f"[RESULT] OCR Numbers: {current_ocr_result}")
            last_api_time = current_time

    # Cooldown message
    elif key in [ord("s"), ord("l"), ord("o")]:
        print(
            f"[WARNING] API Cooldown active. Please wait {API_COOLDOWN - (current_time - last_api_time):.1f} more seconds.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()