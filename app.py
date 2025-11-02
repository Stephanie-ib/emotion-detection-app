import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow import keras
from PIL import Image
import base64
from io import BytesIO

# ================================
# FLASK APP CONFIGURATION
# ================================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ================================
# EMOTION CONFIGURATION
# ================================
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion-specific responses
EMOTION_RESPONSES = {
    'angry': "You look angry. What's bothering you? Take a deep breath! 😤",
    'disgust': "You look disgusted. Did something upset you? 😖",
    'fear': "You seem fearful. Don't worry, everything will be okay! 😨",
    'happy': "You're happy! Keep smiling, it looks great on you! 😊",
    'neutral': "You have a neutral expression. Feeling calm today? 😐",
    'sad': "You look sad. Why are you sad? Cheer up! 😢",
    'surprise': "You look surprised! What amazed you? 😲"
}

MODEL_PATH = 'face_emotionModel.h5'

# ================================
# LOAD TRAINED MODEL
# ================================
print("🔄 Loading emotion detection model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ================================
# DATABASE SETUP
# ================================
def get_db_path():
    """Get database path - use /tmp on Render"""
    if os.environ.get('RENDER'):
        return '/tmp/database.db'
    return 'database.db'

def init_db():
    """Initialize the database and create tables if they don't exist"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if old table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        # Check if old columns exist (email, age, gender)
        cursor.execute("PRAGMA table_info(students)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'email' in columns or 'age' in columns or 'gender' in columns:
            print("⚠️ Old database schema detected. Migrating...")
            # Rename old table
            cursor.execute("ALTER TABLE students RENAME TO students_old")
            
            # Create new table with updated schema
            cursor.execute('''
                CREATE TABLE students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    emotion TEXT,
                    confidence REAL,
                    image_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Copy data from old table (only relevant columns)
            cursor.execute('''
                INSERT INTO students (id, name, emotion, confidence, image_data, timestamp)
                SELECT id, name, emotion, confidence, image_data, timestamp
                FROM students_old
            ''')
            
            # Drop old table
            cursor.execute("DROP TABLE students_old")
            print("✅ Database migrated successfully!")
    else:
        # Create new table
        cursor.execute('''
            CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                emotion TEXT,
                confidence REAL,
                image_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Database created successfully!")
    
    conn.commit()
    conn.close()

# Initialize database when app starts
init_db()

# ================================
# HELPER FUNCTIONS
# ================================
def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))  # Resize to 48x48 (model input size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to 0-1
        img_array = img_array.reshape(1, 48, 48, 1)  # Reshape for model input
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_emotion(image_path):
    """Predict emotion from an image"""
    if model is None:
        return None, 0.0
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, 0.0
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = EMOTIONS[emotion_idx]
        
        return emotion, confidence
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, 0.0

def image_to_base64(image_path):
    """Convert image to base64 string for database storage"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def save_to_database(name, emotion, confidence, image_data):
    """Save student data to database"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO students (name, emotion, confidence, image_data)
            VALUES (?, ?, ?, ?)
        ''', (name, emotion, confidence, image_data))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

# ================================
# ROUTES
# ================================
@app.route('/')
def index():
    """Home page with the form"""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission"""
    try:
        # Get form data
        name = request.form.get('name', '').strip()

        # Validate required fields
        if not name:
            flash('Name is required!', 'error')
            return redirect(url_for('index'))
                
        # Check if image was uploaded
        if 'image' not in request.files:
            flash('No image uploaded!', 'error')
            return redirect(url_for('index'))
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No image selected!', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type! Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Predict emotion
        emotion, confidence = predict_emotion(filepath)
        
        if emotion is None:
            flash('Error detecting emotion. Please try again with a clear face image.', 'error')
            return redirect(url_for('index'))
        
        # Convert image to base64 for database storage
        image_base64 = image_to_base64(filepath)
        
        # Save to database
        success = save_to_database(name, emotion, confidence, image_base64)
        
        if not success:
            flash('Error saving data to database.', 'error')
            return redirect(url_for('index'))
        
        # Get emotion-specific message
        emotion_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
        
        # Create success message
        success_message = f"""
        <h2>✅ Submission Successful!</h2>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Detected Emotion:</strong> {emotion.upper()}</p>
        <p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
        <p><strong>Message:</strong> {emotion_message}</p>
        <img src="/{filepath}" alt="Your uploaded image" style="max-width: 300px; margin-top: 20px; border: 2px solid #333; border-radius: 10px;">
        """
        
        flash(success_message, 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"Error in submit route: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/view_data')
def view_data():
    """View all submitted data (optional - for debugging)"""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, emotion, confidence, timestamp FROM students ORDER BY timestamp DESC')
        data = cursor.fetchall()
        conn.close()
        
        html = "<h1>Submitted Student Data</h1><table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>ID</th><th>Name</th><th>Emotion</th><th>Confidence</th><th>Timestamp</th></tr>"

        for row in data:
            html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]:.2f}%</td><td>{row[4]}</td></tr>"
        
        html += "</table><br><a href='/'>Back to Home</a>"
        return html
    except Exception as e:
        return f"Error: {str(e)}"

# ================================
# RUN APP
# ================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "=" * 60)
    print("🚀 FACE EMOTION DETECTION WEB APP")
    print("=" * 60)
    print("📱 Starting Flask server...")
    print(f"🌐 Access the app at: http://0.0.0.0:{port}")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)