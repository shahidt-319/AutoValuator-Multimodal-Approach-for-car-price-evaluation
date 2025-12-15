import os
import uuid
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, flash, url_for, json ,jsonify
import joblib
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import soundfile as sf
import pandas as pd_yamnet 
import random
import smtplib
from functools import wraps
from flask_mail import Mail, Message
import cv2
import easyocr
from vininfo import Vin
import requests

app = Flask(__name__)
app.secret_key = 'secret123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ADMIN_EMAIL = "shahiddar763312Aa@gmail.com"

STATE_VALIDITY = {
    "Andhra Pradesh": {"Petrol": 15, "Diesel": 15},
    "Arunachal Pradesh": {"Petrol": 15, "Diesel": 15},
    "Assam": {"Petrol": 15, "Diesel": 15},
    "Bihar": {"Petrol": 15, "Diesel": 15},
    "Chhattisgarh": {"Petrol": 15, "Diesel": 15},
    "Goa": {"Petrol": 15, "Diesel": 15},
    "Gujarat": {"Petrol": 15, "Diesel": 15},
    "Haryana": {"Petrol": 15, "Diesel": 15},
    "Himachal Pradesh": {"Petrol": 15, "Diesel": 15},
    "Jharkhand": {"Petrol": 15, "Diesel": 15},
    "Karnataka": {"Petrol": 15, "Diesel": 15},
    "Kerala": {"Petrol": 15, "Diesel": 15},
    "Madhya Pradesh": {"Petrol": 15, "Diesel": 15},
    "Maharashtra": {"Petrol": 15, "Diesel": 15},
    "Manipur": {"Petrol": 15, "Diesel": 15},
    "Meghalaya": {"Petrol": 15, "Diesel": 15},
    "Mizoram": {"Petrol": 15, "Diesel": 15},
    "Nagaland": {"Petrol": 15, "Diesel": 15},
    "Odisha": {"Petrol": 15, "Diesel": 15},
    "Punjab": {"Petrol": 15, "Diesel": 15},
    "Rajasthan": {"Petrol": 15, "Diesel": 15},
    "Sikkim": {"Petrol": 15, "Diesel": 15},
    "Tamil Nadu": {"Petrol": 15, "Diesel": 15},
    "Telangana": {"Petrol": 15, "Diesel": 15},
    "Tripura": {"Petrol": 15, "Diesel": 15},
    "Uttar Pradesh": {"Petrol": 15, "Diesel": 15},
    "Uttarakhand": {"Petrol": 15, "Diesel": 15},
    "West Bengal": {"Petrol": 15, "Diesel": 15},
    # Union Territories
    "Andaman and Nicobar Islands": {"Petrol": 15, "Diesel": 15},
    "Chandigarh": {"Petrol": 15, "Diesel": 15},
    "Dadra and Nagar Haveli and Daman and Diu": {"Petrol": 15, "Diesel": 15},
    "Delhi": {"Petrol": 15, "Diesel": 10},   
    "Jammu and Kashmir": {"Petrol": 25, "Diesel": 25},
    "Ladakh": {"Petrol": 25, "Diesel": 25},
    "Lakshadweep": {"Petrol": 15, "Diesel": 15},
    "Puducherry": {"Petrol": 15, "Diesel": 15},
}
DEFAULT_VALIDITY = {"Petrol": 15, "Diesel": 15}

def get_db():
    conn = sqlite3.connect('autovaluator.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    db = get_db()
    fuel = request.args.get('fuel')
    transmission = request.args.get('transmission')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    modelname = request.args.get('model')

    # Query public details only; skip photo column (will get via car_images)
    query = "SELECT id, car_name, year, fuel, transmission, registration_state, seller_price FROM cars WHERE 1=1"
    params = []
    if fuel:
        query += " AND fuel=?"
        params.append(fuel)
    if transmission:
        query += " AND transmission=?"
        params.append(transmission)
    if min_price:
        query += " AND seller_price >= ?"
        params.append(min_price)
    if max_price:
        query += " AND seller_price <= ?"
        params.append(max_price)
    if modelname:
        query += " AND car_name LIKE ?"
        params.append(f"%{modelname}%")
    query += " ORDER BY id DESC LIMIT 12"
    cars = [dict(row) for row in db.execute(query, params).fetchall()]

    # Attach the main image (or None) for each car from car_images table
    for car in cars:
        img = db.execute("SELECT photo FROM car_images WHERE car_id=? ORDER BY id LIMIT 1", (car['id'],)).fetchone()
        car['photo'] = img['photo'] if img and img['photo'] else None

    ai_models = [
        # ... unchanged ...
        {
            'name': 'Image Damage Detection',
            'desc': 'Detects visual damage from car images using a trained CNN model. Useful for automatic fault detection and market pricing.',
            'long': 'This model classifies uploaded car images into categories like minor, moderate, or severe damage using deep learning. It improves listing accuracy and buyer trust. Trained on thousands of car images with expert-labeled damage, it quickly identifies scratches, dents, and major faults, and integrates damage scores into the price prediction module.',
        },
        {
            'name': 'Audio Engine Fault Detection',
            'desc': 'CNN+GRU hybrid with YAMNet for engine sound classification. Identifies faults based on uploaded engine audio.',
            'long': "Car engine audio is processed using MFCC and embeddings from Google YAMNet. The signal is then classified using a GRU-based deep network as normal, knocking, bearing fault, air leak, and more, providing instant sound-based fault diagnosis.",
        },
        {
            'name': 'VIN Decoding',
            'desc': 'Extracts full car specs using VIN number, leveraging API + ML enhancements.',
            'long': "VIN numbers are decoded by external APIs and ML logic to extract exact make, model, engine details, and manufacturing data. This ensures robust car listing and enables buyers to verify all vehicle history automatically.",
        },
        {
            'name': 'Feature-Based Price Prediction',
            'desc': 'Traditional ML regression model (Random Forest/XGBoost) using car details (year, km, fuel, seller, transmission, location) for price prediction.',
            'long': "When images/audio are missing, our regression model predicts fair resale price using only the car's features. It is trained on hundreds of thousands of curated sale records and produces reproducible, explainable results usable by any seller.",
        }
    ]
    project_features = [
        'Cloud-based AI marketplace for cars',
        'Deep learning and classic ML models used for rich analysis',
        'Instant price prediction and breakdown',
        'OTP secured registration and easy account management',
        'Step-by-step intelligent car listing',
        'Advanced filters (fuel, transmission, price, model)',
        'Add-to-cart, wishlist, compare, and more'
    ]
    half_len = len(project_features) // 2
    project_features_left = project_features[:half_len]
    project_features_right = project_features[half_len:]
    fuel_options = sorted({row[0].capitalize() for row in db.execute("SELECT DISTINCT fuel FROM cars").fetchall() if row[0]})
    transmission_options = sorted({row[0] for row in db.execute("SELECT DISTINCT transmission FROM cars").fetchall() if row[0]})

    return render_template(
        'home.html',
        cars=cars,
        ai_models=ai_models,
        project_features_left=project_features_left,
        project_features_right=project_features_right,
        fuel_options=fuel_options,
        transmission_options=transmission_options
    )

# Routes related to user authentication
@app.route('/')
def index():
    return render_template('index.html')

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='shahiddar763312@gmail.com',
    MAIL_PASSWORD='gdmw wusb dmaq swke'       
)
mail = Mail(app)

def send_otp_email(recipient, otp):
    msg = Message(
        subject="Your AutoValuator OTP",
        recipients=[recipient],
        body=f"Your OTP code is: {otp}",
        sender="shahiddar763312@gmail.com"
    )
    mail.send(msg)

@app.route('/send_otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    method_type = data.get('type')
    email = data.get('email')
    phone = data.get('phone')

    otp = str(random.randint(100000, 999999))
    db = get_db()

    if method_type == 'email' and email:
        db.execute("UPDATE users SET otp=?, otp_verified=0 WHERE email=?", (otp, email))
        db.commit()

        try:
            send_otp_email(email, otp)
            return jsonify({'message': 'OTP sent to email.'})
        except Exception as e:
            return jsonify({'message': f'Failed to send email: {str(e)}'}), 500

    elif method_type == 'phone' and phone:
        db.execute("UPDATE users SET otp=?, otp_verified=0 WHERE phone=?", (otp, phone))
        db.commit()

        print(f"OTP sent to phone {phone}: {otp}")
        return jsonify({'message': 'OTP sent to phone.'})

    return jsonify({'message': 'Invalid request.'}), 400

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    db = get_db()
    email = request.form.get('email')
    phone = request.form.get('phone')
    entered_otp = request.form.get('otp')
    user = None

    if email:
        user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    elif phone:
        user = db.execute("SELECT * FROM users WHERE phone=?", (phone,)).fetchone()

    if user and user['otp'] == entered_otp:
        db.execute("UPDATE users SET otp_verified=1 WHERE id=?", (user['id'],))
        db.commit()
        flash("OTP verified! You can now log in.")
        return redirect('/login') 
    else:
        flash("Invalid OTP. Try again.")
        return redirect('/login')

@app.route('/login', methods=['GET'])
def login_get():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    db = get_db()
    email = request.form.get('email')
    phone = request.form.get('phone')
    password = request.form.get('password')
    user = None

    # Check login via email OR phone
    if email:
        user = db.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password)).fetchone()
    elif phone:
        user = db.execute("SELECT * FROM users WHERE phone=? AND password=?", (phone, password)).fetchone()
    else:
        flash("Provide email or phone to login.")
        return redirect('/')

    if user:
        session['user_id'] = user['id']        # ADD THIS
    session['user'] = user['email']        # email only
    session['role'] = user['role']
    session['is_admin'] = (user['role'] == 'admin')
    flash("Login successful!")
    return redirect('/dashboard')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('index.html', show_register=True)
    db = get_db()
    username = request.form.get('username')
    email = request.form.get('email')
    phone = request.form.get('phone')
    password = request.form.get('password')
    role = request.form.get('role')

    if not username:
        flash("Username is required.")
        return redirect('/')

    if not email and not phone:
        flash("You must provide either an email or phone number.")
        return redirect('/')
    if email and phone:
        flash("Please provide only one: email OR phone.")
        return redirect('/')

    otp = str(random.randint(100000, 999999))

    db.execute("""
        INSERT INTO users (username, email, phone, password, role, otp, otp_verified) 
        VALUES (?, ?, ?, ?, ?, ?, 0)
    """, (username, email, phone, password, role, otp))
    db.commit()

    if email:
        print(f"📧 OTP for {email}: {otp}")
    elif phone:
        print(f"📱 OTP for {phone}: {otp}")

    flash("Registration started. Please verify OTP.")
    return redirect('/login')

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin') and session.get('role') != 'admin':
            flash('Admin access required.')
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/delete-user/<int:user_id>/', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    db = get_db()
    # Delete user's cars and images (cascade delete)
    user_cars = db.execute("SELECT id FROM cars WHERE user=?", (user_id,)).fetchall()
    for car in user_cars:
        db.execute("DELETE FROM car_images WHERE car_id=?", (car['id'],))
        db.execute("DELETE FROM cart WHERE car_id=?", (car['id'],))
    db.execute("DELETE FROM cars WHERE user=?", (user_id,))
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit()
    flash("User and all their cars deleted.")
    return redirect('/admin-panel')

@app.route('/admin/delete-car/<int:car_id>/', methods=['POST'])
@admin_required
def admin_delete_car(car_id):
    db = get_db()
    db.execute("DELETE FROM car_images WHERE car_id=?", (car_id,))
    db.execute("DELETE FROM cart WHERE car_id=?", (car_id,))
    db.execute("DELETE FROM cars WHERE id=?", (car_id,))
    db.commit()
    flash("Car deleted.")
    return redirect('/admin-panel')

# Optional: if you use posts or need to delete other entities
@app.route('/admin/delete-post/<int:post_id>/', methods=['POST'])
@admin_required
def admin_delete_post(post_id):
    db = get_db()
    db.execute("DELETE FROM posts WHERE id=?", (post_id,))
    db.commit()
    flash("Post deleted.")
    return redirect('/admin-panel')

from werkzeug.security import check_password_hash

@app.route('/admin/confirm-delete-user/<int:user_id>/', methods=['GET', 'POST'])
@admin_required
def confirm_delete_user(user_id):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        flash("User not found.")
        return redirect('/admin-panel')

    if request.method == 'POST':
        password = request.form.get('password', '')
        admin = db.execute("SELECT * FROM users WHERE id=?", (session['user_id'],)).fetchone()
        if not admin or admin['password'] != password:  # plain-text password
            flash("Incorrect password. User not deleted.")
            return redirect(f'/admin/confirm-delete-user/{user_id}/')

        user_cars = db.execute("SELECT id FROM cars WHERE user=?", (user_id,)).fetchall()
        for car in user_cars:
            db.execute("DELETE FROM car_images WHERE car_id=?", (car['id'],))
            db.execute("DELETE FROM cart WHERE car_id=?", (car['id'],))
        db.execute("DELETE FROM cars WHERE user=?", (user_id,))
        db.execute("DELETE FROM users WHERE id=?", (user_id,))
        db.commit()
        flash("User and all their cars deleted.")
        return redirect('/admin-panel')

    return render_template('admin_confirm_delete.html', target_type='user', user=user)

@app.route('/admin/confirm-delete-car/<int:car_id>/', methods=['GET', 'POST'])
@admin_required
def confirm_delete_car(car_id):
    db = get_db()
    car = db.execute("SELECT * FROM cars WHERE id=?", (car_id,)).fetchone()
    if not car:
        flash("Car not found.")
        return redirect('/admin-panel')

    if request.method == 'POST':
        password = request.form.get('password', '')
        admin = db.execute("SELECT * FROM users WHERE id=?", (session['user_id'],)).fetchone()
        if not admin or admin['password'] != password:
            flash("Incorrect password. Car not deleted.")
            return redirect(f'/admin/confirm-delete-car/{car_id}/')

        db.execute("DELETE FROM car_images WHERE car_id=?", (car_id,))
        db.execute("DELETE FROM cart WHERE car_id=?", (car_id,))
        db.execute("DELETE FROM cars WHERE id=?", (car_id,))
        db.commit()
        flash("Car deleted.")
        return redirect('/admin-panel')

    return render_template('admin_confirm_delete.html', target_type='car', car=car)

@app.route('/admin-panel')
@admin_required
def admin_panel():
    db = get_db()

    # --- User filter: role ---
    user_role_filter = request.args.get('user_role', '').strip()  # '' or 'buyer'/'seller'/'admin'

    user_query = """
        SELECT MIN(id) as id, email, role 
        FROM users
        WHERE email IS NOT NULL
    """
    user_params = []
    if user_role_filter:
        user_query += " AND role = ?"
        user_params.append(user_role_filter)
    user_query += " GROUP BY email, role"

    users = db.execute(user_query, tuple(user_params)).fetchall()

    # --- Car filters ---
    model_filter = request.args.get('model', '').strip()
    state_filter = request.args.get('state', '').strip()
    min_price = request.args.get('min_price', '').strip()
    max_price = request.args.get('max_price', '').strip()
    fuel_filter = request.args.get('fuel', '').strip()
    year_filter = request.args.get('year', '').strip()
    transmission_filter = request.args.get('transmission', '').strip()

    query = "SELECT * FROM cars WHERE 1=1"
    params = []

    if model_filter:
        query += " AND car_name LIKE ?"
        params.append(f"%{model_filter}%")
    if state_filter:
        query += " AND registration_state = ?"
        params.append(state_filter)
    if min_price:
        query += " AND seller_price >= ?"
        params.append(float(min_price))
    if max_price:
        query += " AND seller_price <= ?"
        params.append(float(max_price))
    if fuel_filter:
        query += " AND fuel = ?"
        params.append(fuel_filter)
    if year_filter:
        query += " AND year = ?"
        params.append(int(year_filter))
    if transmission_filter:
        query += " AND transmission = ?"
        params.append(transmission_filter)

    car_rows = db.execute(query, tuple(params)).fetchall()

    state_rows = db.execute("SELECT DISTINCT registration_state FROM cars WHERE registration_state IS NOT NULL").fetchall()
    all_states = [r['registration_state'] for r in state_rows]

    fuel_rows = db.execute("SELECT DISTINCT fuel FROM cars WHERE fuel IS NOT NULL").fetchall()
    all_fuels = [r['fuel'] for r in fuel_rows]

    trans_rows = db.execute("SELECT DISTINCT transmission FROM cars WHERE transmission IS NOT NULL").fetchall()
    all_transmissions = [r['transmission'] for r in trans_rows]

    year_rows = db.execute("SELECT DISTINCT year FROM cars WHERE year IS NOT NULL ORDER BY year DESC").fetchall()
    all_years = [r['year'] for r in year_rows]

    cars = []
    for car in car_rows:
        photo_row = db.execute(
            "SELECT photo FROM car_images WHERE car_id=? ORDER BY id LIMIT 1",
            (car['id'],)
        ).fetchone()
        car = dict(car)
        car['photo'] = photo_row['photo'] if photo_row else None
        cars.append(car)

    return render_template(
        'admin_panel.html',
        users=users,
        cars=cars,
        all_states=all_states,
        all_fuels=all_fuels,
        all_transmissions=all_transmissions,
        all_years=all_years,
        selected_model=model_filter,
        selected_state=state_filter,
        selected_min_price=min_price,
        selected_max_price=max_price,
        selected_fuel=fuel_filter,
        selected_year=year_filter,
        selected_transmission=transmission_filter,
        selected_user_role=user_role_filter
    )

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    db = get_db()
    cars = []
    # Allow both sellers and admins to see/manage their uploaded cars
    if session['role'] == 'seller' or session['role'] == 'admin':
        cars = db.execute("SELECT * FROM cars WHERE user=?", (session['user'],)).fetchall()
    return render_template(
        'dashboard.html',
        username=session['user'],
        role=session['role'],
        cars=cars
    )


# feature model imports 
model = joblib.load('model.pkl')
columns = joblib.load('model_features.pkl')

# Audio/image model imports
from image_model.predict_damage import predict_damage
from audio_model.audio_preprocess import preprocess_audio
from audio_model.audio_predict import predict_engine_sound_cnn_gru
fuel_classifier = joblib.load('audio_model/fuel_classifier.pkl')

model_path = os.path.abspath("audio_model/audio_cnn_gru_model.h5")

# --- Load CNN+GRU model and label map
cnn_gru_model = tf.keras.models.load_model(model_path)
with open("audio_model/cnn_gru_labels.json") as f:
    label_to_index = json.load(f)

max_index = max(label_to_index.values())
index_to_label_list = [None] * (max_index + 1)
for label, idx in label_to_index.items():
    index_to_label_list[idx] = label

# Load YAMNet model
print("[INFO] Loading YAMNet model...")
yamnet = hub.load('audio_model/yamnet')
class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
yamnet_classes = pd_yamnet.read_csv(class_map_path)['display_name'].to_list()
print(f"[INFO] YAMNet loaded with {len(yamnet_classes)} classes.")

def extract_yamnet_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform)
    return np.mean(embeddings.numpy(), axis=0)

def predict_fuel_type(audio_path):
    emb = extract_yamnet_embedding(audio_path)
    return fuel_classifier.predict([emb])[0]

def pad_mfcc(mfcc, max_frames=100):
    if mfcc.shape[0] > max_frames:
        return mfcc[:max_frames, :]
    else:
        pad_amt = max_frames - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad_amt), (0, 0)), mode='constant')

def predict_fault(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    mfcc_padded = pad_mfcc(mfcc, 100)
    input_array = np.expand_dims(mfcc_padded, axis=0)

    probs = cnn_gru_model.predict(input_array)
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = index_to_label_list[pred_idx] if 0 <= pred_idx < len(index_to_label_list) else "unknown"
    return label.strip().lower(), confidence

# Engine sound extraction function
def extract_engine_and_clipped_yamnet(input_path, engine_output_path, clipped_output_path, threshold=0.1):
    y, sr = librosa.load(input_path, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    scores, _, _ = yamnet(waveform)
    scores_np = scores.numpy()
    try:
        engine_idx = yamnet_classes.index('Engine')
    except ValueError:
        raise RuntimeError("Class 'Engine' not found in YAMNet classes")
    engine_frames = (scores_np[:, engine_idx] > threshold)
    print(f"[YAMNet] Number of frames labeled engine: {np.sum(engine_frames)}")
    frame_hop_sec, frame_len_sec = 0.48, 0.96
    segments_engine = []
    segments_clipped = []

    for i, is_engine in enumerate(engine_frames):
        start_sample = int(i * frame_hop_sec * sr)
        end_sample = int(start_sample + frame_len_sec * sr)
        end_sample = min(end_sample, len(y))
        if is_engine:
            segments_engine.append(y[start_sample:end_sample])
        else:
            segments_clipped.append(y[start_sample:end_sample])
    if segments_engine and np.sum(engine_frames) >= 2:
        engine_audio = np.concatenate(segments_engine)
        sf.write(engine_output_path, engine_audio, sr)
    else:
        sf.write(engine_output_path, y, sr)
    if segments_clipped:
        clipped_audio = np.concatenate(segments_clipped)
        sf.write(clipped_output_path, clipped_audio, sr)
    else:
        sf.write(clipped_output_path, np.zeros(1), sr)
    return True

@app.route('/list-car/basic', methods=['GET', 'POST'])
def list_car_basic():
    if 'user' not in session or session['role'] not in ['seller', 'admin']:
        return redirect('/')
    if request.method == 'POST':
        vin_details_json = request.form.get('vin_details_json', None)
        print("VIN DETAILS JSON received:", vin_details_json)
        session['car_basic'] = {
            'car_name': request.form['car_name'],
            'year': request.form['year'],
            'km_driven': request.form['km_driven'],
            'owner': request.form['owner'],
            'fuel': request.form['fuel'],
            'seller': request.form['seller'],
            'transmission': request.form['transmission'],
            'location': request.form['location'],
            'registration_state': request.form['registration_state'],
            'email': request.form['email'],
            'vin_details': vin_details_json or ''
        }
        return redirect('/list-car/image')
    return render_template('list_car_basic.html', step=1)


vin_reader = easyocr.Reader(['en'], gpu=False)

year_map = {
    'A': 1980, 'B': 1981, 'C': 1982, 'D': 1983, 'E': 1984, 'F': 1985, 'G': 1986, 'H': 1987,
    'J': 1988, 'K': 1989, 'L': 1990, 'M': 1991, 'N': 1992, 'P': 1993, 'R': 1994, 'S': 1995,
    'T': 1996, 'V': 1997, 'W': 1998, 'X': 1999, 'Y': 2000, '1': 2001, '2': 2002, '3': 2003,
    '4': 2004, '5': 2005, '6': 2006, '7': 2007, '8': 2008, '9': 2009, 'A': 2010, 'B': 2011,
    'C': 2012, 'D': 2013, 'E': 2014, 'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
    'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024, 'S': 2025
}
indian_wmi_map = {
    "MA1": "Mahindra & Mahindra",
    "MAT": "Tata Motors",
    "MCA": "Honda Cars India",
    "MLC": "Suzuki Motorcycle India",
    "MA3": "Maruti Suzuki India",
    "MBL": "Hero MotoCorp",
    "MBH": "Bajaj Auto",
    "MBB": "Bajaj Auto",
    "MBC": "Bajaj Auto",
    "MBD": "Bajaj Auto",
    "MBE": "Bajaj Auto",
    "MBF": "Bajaj Auto",
    "MBG": "Bajaj Auto",
    "MBI": "Bajaj Auto",
    "MBJ": "Bajaj Auto",
    "MBM": "Bajaj Auto",
    "MCA": "Honda Cars India",
    "MCB": "Honda Cars India",
    "MCC": "Honda Cars India",
    "MCD": "Honda Cars India",
    "MCE": "Honda Cars India",
    "MFJ": "Honda Motorcycle & Scooter India",
    "MFN": "Honda Motorcycle & Scooter India",
    "MFO": "Honda Motorcycle & Scooter India",
    "MFQ": "Honda Motorcycle & Scooter India",
    "MFU": "Honda Motorcycle & Scooter India",
    "ME1": "Honda Motorcycle & Scooter India",
    "ME2": "Honda Motorcycle & Scooter India",
    "ME3": "Honda Motorcycle & Scooter India",
    "ME4": "Honda Motorcycle & Scooter India",
    "ME5": "Honda Motorcycle & Scooter India",
    "ME6": "Honda Motorcycle & Scooter India",
    "ME7": "Honda Motorcycle & Scooter India",
    "ME8": "Honda Motorcycle & Scooter India",
    "ME9": "Honda Motorcycle & Scooter India",
    "JM3": "Mazda",
    "MH1": "Hyundai",
    "MRH": "Honda",
    "KNA": "Kia Motors",
    "KMH": "Hyundai",
    "KNM": "Nissan",
    "MAL": "Maruti Suzuki India",
    "MMC": "Mitsubishi",
    "MZ1": "Mahindra & Mahindra",
    "PL6": "Toyota",
    "YS2": "Volvo",
}

def correct_vin(text):
    corrections = {"I": "1", "O": "0", "Q": "0", "Z": "2", "S": "5"}
    return "".join(corrections.get(ch, ch) for ch in text.upper() if ch.isalnum())

@app.route('/vin-lookup', methods=['POST'])
def vin_lookup():
    vin = None
    #Get VIN from text or image
    if 'vin' in request.form and request.form['vin']:
        vin = request.form['vin'].strip().upper()
    elif 'vin_image' in request.files:
        vin_image = request.files['vin_image']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vinimg_{os.urandom(6).hex()}.jpg")
        vin_image.save(img_path)
        image = cv2.imread(img_path)
        os.remove(img_path)
        if image is None:
            return jsonify(success=False, error='Could not read uploaded image.')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = vin_reader.readtext(thresh)
        if results:
            results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
            raw_text = "".join([text for (_, text, _) in results])
            vin = correct_vin(raw_text)[:17]
        else:
            return jsonify(success=False, error='No VIN found in image.')
    else:
        return jsonify(success=False, error='No VIN number or image provided.')

    if not vin or len(vin) < 17:
        return jsonify(success=False, error='Failed to read a valid 17-char VIN.')

    try:
        info = Vin(vin)
        year_char = vin[9]
        model_year = year_map.get(year_char.upper(), '')
        wmi = vin[0:3].upper()
        manufacturer = indian_wmi_map.get(wmi, 'Unknown Manufacturer')

        api_url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json"
        api_resp = requests.get(api_url, timeout=5)
        api_data = api_resp.json()
        api_results = api_data.get('Results', [])
        
        api_decoded = {}
        for item in api_results:
            var = item.get('Variable')
            val = item.get('Value')
            if val and val != 'Not Applicable' and val != 'Not Available' and val != '':
                api_decoded[var] = val

        result_data = {
            'vin': vin,
            'car_name': manufacturer,
            'year': model_year,
            'fuel': api_decoded.get('Fuel Type - Primary', getattr(info, 'fuel_type', '')),
            'transmission': api_decoded.get('Transmission Style', getattr(info, 'transmission_type', '')),
            'body_class': api_decoded.get('Body Class', ''),
            'engine_model': api_decoded.get('Engine Model', ''),
            'engine_cylinders': api_decoded.get('Engine Number of Cylinders', ''),
            'drive_type': api_decoded.get('Drive Type', ''),
            'plant_country': api_decoded.get('Plant Country', ''),
            'plant_city': api_decoded.get('Plant City', ''),
            'air_bags': api_decoded.get('Number of Airbags', ''),
            'engine_displacement_cc': api_decoded.get('Displacement (CC)', ''),
            'vehicle_type': api_decoded.get('Vehicle Type', ''),
            'trim': api_decoded.get('Trim', ''),
            'model': api_decoded.get('Model', ''),
            'make': api_decoded.get('Make', ''),
            'series': api_decoded.get('Series', ''),
            'registration_state': '' 
        }

        return jsonify(success=True, **result_data)

    except Exception as e:
        return jsonify(success=False, error='Error decoding VIN: ' + str(e))
@app.route('/list-car/image', methods=['GET', 'POST'])
def list_car_image():
    if 'user' not in session or session['role'] not in ['seller', 'admin']:
        return redirect('/')
    if 'car_basic' not in session:
        return redirect('/list-car/basic')

    if request.method == 'POST':
        # AJAX detection: receive multiple images, return list of results
        if 'detect_damage' in request.form:
            photos = request.files.getlist('photos[]')
            if not photos or not any(p.filename for p in photos):
                return jsonify({'error': 'No photos uploaded.'}), 400

            results = []
            for photo in photos:
                if not photo or not photo.filename:
                    results.append({'filename': '', 'damage_status': 'unknown'})
                    continue
                ext = os.path.splitext(photo.filename)[1]
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4().hex}{ext}")
                photo.save(temp_path)
                try:
                    damage_status, _ = predict_damage(temp_path)
                except Exception:
                    damage_status = "unknown"
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                results.append({'filename': photo.filename, 'damage_status': damage_status})

            return jsonify({'results': results})

        # Final submit: accept multiple files (but through normal multi-upload post)
        elif 'submit' in request.form:
            files = request.files.getlist('photos')
            if not files or not any(f.filename for f in files):
                flash("Please upload at least one photo.")
                return redirect('/list-car/image')

            image_names = []
            damage_list = []
            for file in files:
                if not file or not file.filename:
                    continue
                ext = os.path.splitext(file.filename)[1]
                unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{ext}"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                file.save(save_path)
                try:
                    damage_status, _ = predict_damage(save_path)
                except Exception:
                    damage_status = "unknown"
                image_names.append(unique_name)
                damage_list.append(damage_status)
            
            session['car_images'] = image_names   # List of filenames
            session['damage_status_list'] = damage_list # List of detected damages

            # Update car_basic in session with form fields (if needed)
            cb = session.get('car_basic', {}).copy()
            cb['car_name'] = request.form.get('car_name')
            cb['year'] = request.form.get('year')
            cb['km_driven'] = request.form.get('km_driven')
            cb['owner'] = request.form.get('owner')
            cb['fuel'] = request.form.get('fuel')
            cb['seller'] = request.form.get('seller')
            cb['transmission'] = request.form.get('transmission')
            cb['location'] = request.form.get('location')
            cb['registration_state'] = request.form.get('registration_state')
            cb['email'] = request.form.get('email')
            session['car_basic'] = cb 

            return redirect('/list-car/audio')

    return render_template('list_car_image.html', step=2)

@app.route('/list-car/audio', methods=['GET', 'POST'])
def list_car_audio():
    if 'user' not in session or session['role'] not in ['seller', 'admin']:
        return redirect('/')
    # Make sure all session keys exist before GET or POST
    if 'car_basic' not in session or 'car_images' not in session or 'damage_status_list' not in session:
        return redirect('/list-car/basic')

    car_basic = session['car_basic']
    car_name = car_basic['car_name']
    year = car_basic['year']
    km_driven = car_basic['km_driven']
    fuel = car_basic['fuel']
    seller = car_basic['seller']
    transmission = car_basic['transmission']
    registration_state = car_basic['registration_state']
    photo_filenames = session['car_images']
    vin_details_json = car_basic.get('vin_details', None)

    engine_condition = session.get('engine_condition', '')
    damage_status_list = session.get('damage_status_list', ['Unknown'])
    cleaned_audio_url = None
    predicted_price = None
    fuel_type_predicted = None
    fuel_mismatch_warning = None

    if request.method == 'POST':
        audio = request.files.get('engine_audio')
        if not audio or not audio.filename:
            flash('Please upload engine audio.')
            return render_template(
                'list_car_audio.html',
                car_name=car_name,
                year=year,
                km_driven=km_driven,
                fuel=fuel,
                seller=seller,
                transmission=transmission,
                registration_state=registration_state,
                photo_filenames=photo_filenames,
                engine_condition=engine_condition,
                damage_status_list=damage_status_list,
                cleaned_audio_url=None,
                predicted_price=None,
                step=3
            )

        audio_ext = os.path.splitext(audio.filename)[1]
        original_audio_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{audio_ext}"
        original_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], original_audio_filename)
        audio.save(original_audio_path)

        cleaned_audio_filename = original_audio_filename.rsplit('.', 1)[0] + "_cleaned.wav"
        noise_audio_filename = original_audio_filename.rsplit('.', 1)[0] + "_noise.wav"
        clipped_audio_filename = original_audio_filename.rsplit('.', 1)[0] + "_clipped.wav"
        cleaned_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_audio_filename)
        noise_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], noise_audio_filename)
        clipped_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], clipped_audio_filename)

        try:
            extract_engine_and_clipped_yamnet(original_audio_path, cleaned_audio_path, clipped_audio_path, threshold=0.1)
            fuel_type_predicted = predict_fuel_type(cleaned_audio_path)
            engine_condition, confidence = predict_fault(cleaned_audio_path)
            engine_factor_map = {
                "air leak": 0.90, "air leak engine inside cabin": 0.88, "background noise": 0.95,
                "bearing fault": 0.6, "idling": 0.97, "knocking": 0.7, "normal engine inside cabin": 1.0,
                "oil cap off engine inside cabin": 0.85, "oil leak noise": 0.88
            }
            engine_factor = engine_factor_map.get(engine_condition, 1.0)
            cleaned_audio_url = url_for('static', filename=f'uploads/{cleaned_audio_filename}')
        except Exception as e:
            flash(f"Error processing audio: {e}")
            engine_condition = "unknown"
            engine_factor = 1.0
            cleaned_audio_url = None
            fuel_type_predicted = None
        finally:
            if os.path.exists(original_audio_path):
                os.remove(original_audio_path)

        try:
            seller_price = float(request.form.get('seller_price', 0))
        except ValueError:
            flash("Invalid seller price.")
            seller_price = 0

        fuel_user = car_basic['fuel'].lower() if car_basic['fuel'] else None
        if fuel_type_predicted and fuel_user and fuel_type_predicted.lower() != fuel_user:
            fuel_mismatch_warning = (
                f"Warning: Your audio indicates fuel type '{fuel_type_predicted}', which does not match "
                f"your selected fuel type '{fuel_user}'. The selected fuel type will be updated."
            )
            car_basic['fuel'] = fuel_type_predicted
            fuel = fuel_type_predicted

        year_int = int(year)
        age = datetime.now().year - year_int
        km_int = int(km_driven)
        damage_levels = {'severe': 0.75, 'moderate': 0.85, 'minor': 0.95, 'unknown': 1.0}
        image_damage_factors = [damage_levels.get(ds.lower(), 1.0) for ds in damage_status_list]
        damage_factor = min(image_damage_factors) if image_damage_factors else 1.0

        input_data = pd.DataFrame([{
            'Age': age, 'Kms_Driven': km_int, 'Fuel_Type': fuel,
            'Seller_Type': seller, 'Transmission': transmission, 'Car_Name': car_name
        }])[columns]

        base_price = model.predict(input_data)[0]
        validity_years = STATE_VALIDITY.get(registration_state, DEFAULT_VALIDITY).get(fuel, 15)
        remaining_years = validity_years - age
        state_factor = 0.2 if remaining_years <= 0 else max(0.3, remaining_years / validity_years)
        final_price = round(base_price * damage_factor * engine_factor * state_factor * 2, 2)
        predicted_price = final_price

        # --- Actual DB Write: cars and car_images ---
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            INSERT INTO cars
            (user, year, km_driven, owner, fuel, seller, transmission, location,
             registration_state, engine_condition, email, car_name, predicted_price,
             seller_price, cleaned_audio, noise_audio, clipped_audio, vin_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session['user'], year, km_driven, car_basic['owner'],
                fuel, seller, transmission, car_basic['location'], registration_state,
                engine_condition, car_basic['email'], car_name,
                predicted_price, seller_price, cleaned_audio_filename, noise_audio_filename,
                clipped_audio_filename, vin_details_json
            )
        )
        car_id = cur.lastrowid
        for fname, status in zip(photo_filenames, damage_status_list):
            cur.execute(
                "INSERT INTO car_images (car_id, photo, damage_status) VALUES (?, ?, ?)",
                (car_id, fname, status)
            )
        db.commit()

        session.pop('car_images', None)
        session.pop('damage_status_list', None)
        session.pop('car_basic', None)
        session.pop('engine_condition', None)
        session.pop('damage_status', None)

        if fuel_mismatch_warning:
            flash(fuel_mismatch_warning)

        # --- REDIRECT to price breakdown step instead of render ---
        return redirect(url_for('price_breakdown', car_id=car_id))

    # Default GET
    return render_template(
        'list_car_audio.html',
        car_name=car_name,
        year=year,
        km_driven=km_driven,
        fuel=fuel,
        seller=seller,
        transmission=transmission,
        registration_state=registration_state,
        photo_filenames=photo_filenames,
        engine_condition=engine_condition,
        damage_status_list=damage_status_list,
        cleaned_audio_url=None,
        predicted_price=None,
        predicted_fuel=None,
        step=3
    )


@app.route('/api/predict-price', methods=['POST'])
def api_predict_price():
    if 'user' not in session or session['role'] not in ['seller', 'admin']:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json(force=True)

    required_fields = ['year', 'km_driven', 'fuel', 'seller', 'transmission', 'car_name', 'registration_state']
    if not all(field in data and data[field] for field in required_fields):
        return jsonify({'error': 'Missing fields'}), 400

    try:
        year = int(data['year'])
        km = int(data['km_driven'])
        fuel = data['fuel']
        seller = data['seller']
        transmission = data['transmission']
        car_name = data['car_name']
        registration_state = data['registration_state']

        age = datetime.now().year - year
        input_data = pd.DataFrame([{
            'Age': age,
            'Kms_Driven': km,
            'Fuel_Type': fuel,
            'Seller_Type': seller,
            'Transmission': transmission,
            'Car_Name': car_name
        }])[columns]

        base_price = model.predict(input_data)[0]

        validity = STATE_VALIDITY.get(registration_state, DEFAULT_VALIDITY)
        state_years = validity.get(fuel, 15)
        remaining_years = state_years - age
        state_factor = 0.2 if remaining_years <= 0 else max(0.3, remaining_years / state_years)

        predicted_price = round(base_price * state_factor * 2, 2)

        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        return jsonify({'error': f'Error computing prediction: {str(e)}'}), 500

@app.route('/predict-image-audio', methods=['POST'])
def predict_image_audio():
    form = request.form
    year = int(form['year'])
    age = datetime.now().year - year
    km = int(form['km_driven'])
    fuel = form['fuel']
    seller = form['seller']
    transmission = form['transmission']
    car_name = form['car_name']
    registration_state = form.get('registration_state', '')

    photos = request.files.getlist('photos[]') or []
    audio = request.files.get('engine_audio')

    # Image damage prediction(s)
    damage_statuses = []
    for photo in photos:
        if photo and photo.filename:
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4().hex}{os.path.splitext(photo.filename)[1]}")
            photo.save(photo_path)
            try:
                damage_status, _ = predict_damage(photo_path)
            except Exception as e:
                print(f"Image AI error: {e}")
                damage_status = "unknown"
            finally:
                os.remove(photo_path)
            damage_statuses.append(damage_status)
        else:
            damage_statuses.append("unknown")

    # Robust damage summary logic for full-car rating:
    priority = {'severe': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
    distinct_types = set(ds.lower() for ds in damage_statuses)
    if len(distinct_types) > 1 and 'unknown' not in distinct_types:
        overall_damage = 'unknown'
    else:
        # Pick worst of present (severe > moderate > minor > unknown)
        overall_damage = max(damage_statuses, key=lambda d: priority.get(d.lower(), 0), default='unknown')

    damage_levels = {'minor': 0.95, 'moderate': 0.85, 'severe': 0.75, 'unknown': 1.0}
    damage_factor = damage_levels.get(overall_damage.lower(), 1.0)

    cleaned_audio_url = None
    cleaned_audio_fname = None
    engine_condition, engine_factor = 'unknown', 1.0

    # Audio: (unchanged from your version)
    if audio and audio.filename:
        audio_ext = os.path.splitext(audio.filename)[1]
        uid = uuid.uuid4().hex
        raw_audio_fname = f"temp_{uid}{audio_ext}"
        raw_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_audio_fname)
        audio.save(raw_audio_path)
        cleaned_audio_fname = f"pred_{uid}_cleaned.wav"
        cleaned_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_audio_fname)
        try:
            extract_engine_and_clipped_yamnet(raw_audio_path, cleaned_audio_path, cleaned_audio_path)
            y, sr = librosa.load(cleaned_audio_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
            mfcc_padded = pad_mfcc(mfcc, 100)
            input_array = np.expand_dims(mfcc_padded, axis=0)
            probs = cnn_gru_model.predict(input_array)
            pred_idx = int(np.argmax(probs))
            raw_label = index_to_label_list[pred_idx] if 0 <= pred_idx < len(index_to_label_list) else "unknown"
            engine_condition = raw_label.strip().lower() if raw_label != "unknown" else "unknown"
            confidence = float(np.max(probs))
        except Exception as e:
            print(f"Audio AI error: {e}")
            engine_condition = 'unknown'
        finally:
            if os.path.exists(raw_audio_path):
                os.remove(raw_audio_path)
        engine_factor = {
            "air leak": 0.90, "air leak engine inside cabin": 0.88, "background noise": 0.95,
            "bearing fault": 0.6, "idling": 0.97, "knocking": 0.7, "normal engine inside cabin": 1.0,
            "oil cap off engine inside cabin": 0.85, "oil leak noise": 0.88
        }.get(engine_condition, 1.0)
        cleaned_audio_url = url_for('static', filename=f'uploads/{cleaned_audio_fname}') if cleaned_audio_fname else None
    else:
        engine_condition = session.get('engine_condition', 'unknown')
        engine_factor = {
            "air leak": 0.90, "air leak engine inside cabin": 0.88, "background noise": 0.95,
            "bearing fault": 0.6, "idling": 0.97, "knocking": 0.7, "normal engine inside cabin": 1.0,
            "oil cap off engine inside cabin": 0.85, "oil leak noise": 0.88
        }.get(engine_condition, 1.0)

    input_data = pd.DataFrame([{
        'Age': age, 'Kms_Driven': km, 'Fuel_Type': fuel,
        'Seller_Type': seller, 'Transmission': transmission, 'Car_Name': car_name
    }])[columns]
    base_price = model.predict(input_data)[0]
    validity_years = STATE_VALIDITY.get(registration_state, DEFAULT_VALIDITY).get(fuel, 15)
    remaining_years = validity_years - age
    state_factor = 0.2 if remaining_years <= 0 else max(0.3, remaining_years / validity_years)
    final_price = round(base_price * damage_factor * engine_factor * state_factor, 2)

    # Save for session
    session['engine_condition'] = engine_condition
    session['damage_status_list'] = damage_statuses
    session['cleaned_audio'] = cleaned_audio_fname if cleaned_audio_url else None
    session.modified = True

    return {
        'predicted_price': final_price,
        'damage_statuses': damage_statuses,
        'overall_damage': overall_damage,
        'engine_condition': engine_condition,
        'cleaned_audio_url': cleaned_audio_url
    }

import json
from datetime import datetime
from flask import session, redirect, flash, request, render_template

@app.route('/list-car/price-breakdown', methods=['GET', 'POST'])
def price_breakdown():
    if 'user' not in session or session['role'] not in ['seller', 'admin']:
        return redirect('/')

    if request.method == 'POST':
        car_id = request.form.get('car_id', type=int)
    else:
        car_id = request.args.get('car_id', type=int)

    if not car_id:
        flash("Missing car ID. Please complete the listing steps.")
        return redirect('/list-car/basic')

    db = get_db()
    car = db.execute("SELECT * FROM cars WHERE id=? AND user=?", (car_id, session['user'])).fetchone()
    if not car:
        flash("Cannot find your car listing. Please start again.")
        return redirect('/list-car/basic')

    images = db.execute("SELECT photo, damage_status FROM car_images WHERE car_id=?", (car_id,)).fetchall()
    photo_filenames = [img['photo'] for img in images]
    damage_status_list = [img['damage_status'] for img in images]

    # Always use the worst damage present (e.g., 'severe' if any).
    priority = {'severe': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
    worst_damage_label = max(damage_status_list, key=lambda d: priority.get(d.lower(), 0), default='unknown')
    damage_levels = {'severe': 0.75, 'moderate': 0.85, 'minor': 0.95, 'unknown': 1.0}
    worst_damage_factor = damage_levels.get(worst_damage_label.lower(), 1.0)

    year = int(car['year'])
    age = datetime.now().year - year
    km = int(car['km_driven'])
    fuel = car['fuel']
    seller = car['seller']
    transmission = car['transmission']
    car_name = car['car_name']
    registration_state = car['registration_state']
    engine_condition = car['engine_condition']

    with open('current_new_car_prices.json', 'r', encoding='utf-8') as f:
        current_prices = json.load(f)
    car_name_key = car_name.strip().lower()
    current_price = current_prices.get(car_name_key, None)
    starting_price = float(current_price or 1000000)

    factors = []
    factors.append([
        'Current New Car Price',
        'Reference market price for 2025',
        f"{starting_price:.2f}"
    ])
    validity_years = STATE_VALIDITY.get(registration_state, DEFAULT_VALIDITY).get(fuel, 15)
    years_used = age
    remaining_years = validity_years - years_used
    state_factor = 0.2 if remaining_years <= 0 else max(0.3, remaining_years / validity_years)
    after_year_price = round(starting_price * state_factor, 2)
    year_deduction = round(starting_price - after_year_price, 2)
    factors.append([
        'Depreciation for Age',
        f'Year: {year} | Used: {years_used} yrs | x{state_factor:.2f} <br><span class="deduct-amount">-₹{year_deduction}</span>',
        f"{after_year_price:.2f}"
    ])
    km_threshold = 50000
    km_unit = 10000
    km_factor_step = 0.01
    extra_km = max(0, km - km_threshold)
    km_factor = 1 - (extra_km // km_unit) * km_factor_step
    km_factor = max(km_factor, 0.85)
    after_km_price = round(after_year_price * km_factor, 2)
    km_deduction = round(after_year_price - after_km_price, 2)
    factors.append([
        'KM Driven Deduction',
        f'{km} km | x{km_factor:.3f} <br><span class="deduct-amount">-₹{km_deduction}</span>',
        f"{after_km_price:.2f}"
    ])
    fuel_factor_map = {'petrol': 1, 'diesel': 0.98, 'cng': 0.96, 'electric': 1.03}
    fuel_factor = fuel_factor_map.get(fuel.lower(), 1)
    after_fuel_price = round(after_km_price * fuel_factor, 2)
    fuel_deduction = round(after_km_price - after_fuel_price, 2)
    factors.append([
        'Fuel Type Adjustment',
        f'{fuel.title()} | x{fuel_factor} <br><span class="deduct-amount">-₹{fuel_deduction}</span>',
        f"{after_fuel_price:.2f}"
    ])
    seller_factor_map = {'dealer': 0.97, 'individual': 1}
    seller_factor = seller_factor_map.get(seller.lower(), 1)
    after_seller_price = round(after_fuel_price * seller_factor, 2)
    seller_deduction = round(after_fuel_price - after_seller_price, 2)
    factors.append([
        'Seller Type Adjustment',
        f'{seller.title()} | x{seller_factor} <br><span class="deduct-amount">-₹{seller_deduction}</span>',
        f"{after_seller_price:.2f}"
    ])
    trans_factor_map = {'manual': 1, 'automatic': 1.03}
    trans_factor = trans_factor_map.get(transmission.lower(), 1)
    after_trans_price = round(after_seller_price * trans_factor, 2)
    trans_deduction = round(after_seller_price - after_trans_price, 2)
    factors.append([
        'Transmission Adjustment',
        f'{transmission.title()} | x{trans_factor} <br><span class="deduct-amount">-₹{trans_deduction}</span>',
        f"{after_trans_price:.2f}"
    ])
    after_damage_price = round(after_trans_price * worst_damage_factor, 2)
    damage_deduction = round(after_trans_price - after_damage_price, 2)
    factors.append([
        'Damage Deduction',
        f'{worst_damage_label.title()} | x{worst_damage_factor:.2f} <br><span class="deduct-amount">-₹{damage_deduction}</span>',
        f"{after_damage_price:.2f}"
    ])
    engine_factor_map = {
        "air leak": 0.90, "air leak engine inside cabin": 0.88, "background noise": 0.95,
        "bearing fault": 0.6, "idling": 0.97, "knocking": 0.7, "normal engine inside cabin": 1.0,
        "oil cap off engine inside cabin": 0.85, "oil leak noise": 0.88
    }
    engine_factor_applied = engine_factor_map.get(str(engine_condition).lower(), 1.0)
    final_price = round(after_damage_price * engine_factor_applied, 2)
    engine_deduction = round(after_damage_price - final_price, 2)
    factors.append([
        'Engine Sound/A.I. Deduction',
        f"{engine_condition.title() if engine_condition else 'unknown'} | x{engine_factor_applied} <br><span class='deduct-amount'>-₹{engine_deduction}</span>",
        f"{final_price:.2f}"
    ])

    seller_price_default = current_price if current_price is not None else final_price

    if request.method == 'POST':
        seller_price = request.form.get('seller_price', type=float)
        db.execute("UPDATE cars SET seller_price=? WHERE id=? AND user=?", (seller_price, car_id, session['user']))
        db.commit()
        flash(f"Car listing completed. Final Predicted Price: ₹{final_price} | Seller Price: ₹{seller_price:.2f}")
        return redirect('/marketplace')

    return render_template(
        'list_car_price_breakdown.html',
        price_factors=factors,
        predicted_price=final_price,
        seller_price=seller_price_default,
        current_price=current_price,
        model_year=year,
        car_model_name=car_name,
        final_price=final_price,
        engine_condition=engine_condition,
        damage_status_list=damage_status_list,
        photo_filenames=photo_filenames,
        car_id=car_id,
        step=4
    )

@app.route('/car/<int:car_id>/')
def car_detail(car_id):
    """Display detailed info for a single car listing, including AI predictions, audio, and VIN details."""
    if 'user' not in session:
        return redirect('/')

    db = get_db()
    try:
        car = db.execute("SELECT * FROM cars WHERE id = ?", (car_id,)).fetchone()
        if car is None:
            flash("Car not found or has been removed.", "warning")
            return redirect('/marketplace')

        car_dict = dict(car)
        # VIN info
        vin_json = car_dict.get('vin_details')
        vin_data = None
        if vin_json:
            try:
                vin_data = json.loads(vin_json)
            except (json.JSONDecodeError, TypeError):
                vin_data = None
        car_dict['vin_details'] = vin_data

        # Load all images and their damage statuses
        images = db.execute("SELECT photo, damage_status FROM car_images WHERE car_id=?", (car_id,)).fetchall()
        car_dict['images'] = [dict(img) for img in images]

        # Set summary damage assessment as the worst type present (including multi-image)
        priority = {'severe': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
        if car_dict['images']:
            worst_damage = max(car_dict['images'], key=lambda img: priority.get((img['damage_status'] or 'unknown').lower(), 0))['damage_status']
            car_dict['damage_status'] = worst_damage.title()
        else:
            car_dict['damage_status'] = 'Not Available'

        # Market price
        current_price_2025 = None
        try:
            with open('current_new_car_prices.json', 'r', encoding='utf-8') as f:
                market_prices = json.load(f)
            model_key = car_dict.get('car_name', '').strip().lower()
            current_price_2025 = market_prices.get(model_key)
        except FileNotFoundError:
            print("⚠️ current_new_car_prices.json not found — skipping market price lookup.")
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON format in current_new_car_prices.json.")

        # Cart check
        in_cart = db.execute(
            "SELECT 1 FROM cart WHERE user = ? AND car_id = ?",
            (session['user'], car_id)
        ).fetchone() is not None

    finally:
        db.close()

    return render_template(
        'car_detail.html',
        car=car_dict,
        current_price_2025=current_price_2025 or "N/A",
        in_cart=in_cart
    )

@app.route('/add-to-cart/<int:car_id>/', methods=['POST'])
def add_to_cart(car_id):
    if 'user' not in session:
        return redirect('/')
    db = get_db()
    car = db.execute("SELECT * FROM cars WHERE id=?", (car_id,)).fetchone()
    if not car:
        return "Car not found", 404
    if db.execute("SELECT * FROM cart WHERE user=? AND car_id=?", (session['user'], car_id)).fetchone():
        return redirect('/cart')
    db.execute("""
        INSERT INTO cart (user, car_id, car_name, fuel, transmission, location, photo, seller_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
               (session['user'], car['id'], car['car_name'], car['fuel'], car['transmission'],
                car['location'], car['photo'], car['seller_price']))
    db.commit()
    return redirect('/cart')

@app.route('/remove-from-cart/<int:car_id>/', methods=['POST'])
def remove_from_cart(car_id):
    if 'user' not in session:
        return redirect('/')
    db = get_db()
    db.execute("DELETE FROM cart WHERE user=? AND car_id=?", (session['user'], car_id))
    db.commit()
    return redirect('/cart')

@app.route('/cart')
def cart():
    if 'user' not in session:
        return redirect('/')
    db = get_db()
    cart_rows = db.execute("SELECT * FROM cart WHERE user=?", (session['user'],)).fetchall()

    def select_worst_damage_image(images):
        priority = {'severe': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
        if not images:
            return None
        return max(images, key=lambda img: priority.get((img.get('damage_status') or 'unknown').lower(), 0))

    items = []
    for row in cart_rows:
        car_id = row['car_id']
        car = db.execute("SELECT * FROM cars WHERE id=?", (car_id,)).fetchone()
        if car is None:
            continue
        car_dict = dict(car)
        # Merge in cart-row values if needed
        car_dict.update(dict(row))
        images = db.execute("SELECT photo, damage_status FROM car_images WHERE car_id=?", (car_id,)).fetchall()
        images = [dict(img) for img in images]
        car_dict['worst_image'] = select_worst_damage_image(images)
        car_dict['car_id'] = car_id  # Needed for remove-from-cart link
        items.append(car_dict)

    return render_template('cart.html', items=items)

@app.route('/my-store')
def my_store():
    if 'user' not in session or session['role'] not in ['seller', 'admin']:
        return redirect('/')
    db = get_db()
    db_cars = db.execute("SELECT * FROM cars WHERE user=?", (session['user'],)).fetchall()

    def select_worst_damage_image(images):
        priority = {'severe': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
        if not images:
            return None
        return max(images, key=lambda img: priority.get((img.get('damage_status') or 'unknown').lower(), 0))

    cars = []
    for row in db_cars:
        car = dict(row)
        images = db.execute("SELECT photo, damage_status FROM car_images WHERE car_id=?", (car['id'],)).fetchall()
        images = [dict(img) for img in images]
        car['worst_image'] = select_worst_damage_image(images)
        cars.append(car)

    return render_template('my_store.html', cars=cars)


@app.route('/remove-from-store/<int:car_id>/', methods=['POST'])
def remove_from_store(car_id):
    if 'user' not in session:
        return redirect('/')
    db = get_db()
    db.execute("DELETE FROM cars WHERE id=? AND user=?", (car_id, session['user']))
    db.commit()
    return redirect('/my-store')

@app.route('/about')
def about():
    return render_template('about_us.html')
 
import json

@app.route('/marketplace')
def marketplace():
    if 'user' not in session:
        return redirect('/')
    db = get_db()

    # Get filter values from query string
    fuel = request.args.get('fuel', '').strip()
    transmission = request.args.get('transmission', '').strip()
    model_name = request.args.get('model', '').strip()
    price = request.args.get('price', '').strip()
    registration_state = request.args.get('registration_state', '').strip()

    # Build query dynamically
    query = "SELECT * FROM cars WHERE 1=1"
    params = []
    if fuel:
        query += " AND fuel = ?"
        params.append(fuel)
    if transmission:
        query += " AND transmission = ?"
        params.append(transmission)
    if model_name:
        query += " AND car_name LIKE ?"
        params.append(f"%{model_name}%")
    if price:
        query += " AND seller_price <= ?"
        params.append(float(price))
    if registration_state:
        query += " AND registration_state = ?"
        params.append(registration_state)

    db_results = db.execute(query, params).fetchall()
    cars = []

    def select_worst_damage_image(images):
        priority = {'severe': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
        return max(
            images,
            key=lambda img: priority.get((img['damage_status'] or 'unknown').lower(), 0),
            default=None
        )

    for row in db_results:
        car = dict(row)
        vin_json = car.get('vin_details')
        if vin_json:
            try:
                car['vin'] = json.loads(vin_json)
            except Exception:
                car['vin'] = None
        else:
            car['vin'] = None

        car_id = car['id']
        images = db.execute(
            "SELECT photo, damage_status FROM car_images WHERE car_id=?",
            (car_id,)
        ).fetchall()
        car['images'] = images
        car['worst_image'] = select_worst_damage_image(images)
        cars.append(car)

    # Distinct values for dropdowns
        # Distinct values for dropdowns (normalized + deduplicated)
    model_rows = db.execute(
        "SELECT DISTINCT car_name FROM cars WHERE car_name IS NOT NULL"
    ).fetchall()
    all_models = sorted({(r['car_name'] or '').strip() for r in model_rows})

    fuel_rows = db.execute(
        "SELECT DISTINCT fuel FROM cars WHERE fuel IS NOT NULL"
    ).fetchall()
    all_fuels = sorted({(r['fuel'] or '').strip().lower() for r in fuel_rows})

    trans_rows = db.execute(
        "SELECT DISTINCT transmission FROM cars WHERE transmission IS NOT NULL"
    ).fetchall()
    all_transmissions = sorted({(r['transmission'] or '').strip().lower() for r in trans_rows})

    state_rows = db.execute(
        "SELECT DISTINCT registration_state FROM cars WHERE registration_state IS NOT NULL"
    ).fetchall()
    all_states = sorted({(r['registration_state'] or '').strip() for r in state_rows})

    with open('current_new_car_prices.json', 'r', encoding='utf-8') as f:
        market_prices = json.load(f)
    for car in cars:
        model_key = (car.get('car_name') or '').strip().lower()
        car['current_price_2025'] = market_prices.get(model_key)

    return render_template(
        'marketplace.html',
        cars=cars,
        all_models=all_models,
        all_fuels=all_fuels,
        all_transmissions=all_transmissions,
        all_states=all_states,
        selected_model=model_name,
        selected_fuel=fuel,
        selected_transmission=transmission,
        selected_state=registration_state,
        selected_price=price
    )

if __name__ == '__main__':
    app.run(debug=True)