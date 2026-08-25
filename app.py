import os

# =========================================================
# RENDER / CPU ONLY CONFIG
# MUST COME BEFORE TENSORFLOW / DEEPFACE IMPORTS
# =========================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sqlite3
import traceback

import cv2
import numpy as np
import tensorflow as tf

# Explicitly tell TensorFlow to use CPU
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from deepface import DeepFace
from werkzeug.security import generate_password_hash, check_password_hash


from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from deepface import DeepFace


app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "emotivision-development-secret"
)

CORS(app)


# =========================================================
# OPENCV CASCADE
# =========================================================

CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cv2_data",
    "haarcascade_frontalface_default.xml"
)

if os.path.exists(CASCADE_PATH):
    cv2.data.haarcascades = (
        os.path.dirname(CASCADE_PATH) + os.sep
    )


# =========================================================
# DATABASE
# =========================================================

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "users.db"
)


def get_db():
    return sqlite3.connect(DB_PATH)


def init_auth_db():
    conn = get_db()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


init_auth_db()


# =========================================================
# PAGES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/features")
def features_page():
    return render_template("features.html")


@app.route("/how-it-works")
def how_it_works_page():
    return render_template("how_it_works.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/blog")
def blog_page():
    return render_template("blog.html")


@app.route("/auth")
def auth_page():
    return render_template("auth.html")


# =========================================================
# AUTH
# =========================================================

@app.route("/api/signup", methods=["POST"])
def signup():

    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or not password:
        return jsonify({
            "success": False,
            "error": "Name, email and password are required."
        }), 400

    if len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Password must contain at least 6 characters."
        }), 400

    try:

        conn = get_db()

        conn.execute(
            """
            INSERT INTO users
            (name, email, password)
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                generate_password_hash(password)
            )
        )

        conn.commit()
        conn.close()

        session["user_name"] = name
        session["user_email"] = email

        return jsonify({
            "success": True,
            "message": "Account created successfully.",
            "name": name
        })

    except sqlite3.IntegrityError:

        return jsonify({
            "success": False,
            "error": "This email is already registered."
        }), 409

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": "Unable to create account."
        }), 500


@app.route("/api/login", methods=["POST"])
def login():

    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({
            "success": False,
            "error": "Email and password are required."
        }), 400

    conn = get_db()

    row = conn.execute(
        """
        SELECT name, email, password
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    conn.close()

    if not row or not check_password_hash(row[2], password):

        return jsonify({
            "success": False,
            "error": "Invalid email or password."
        }), 401

    session["user_name"] = row[0]
    session["user_email"] = row[1]

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "name": row[0]
    })


@app.route("/api/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "success": True
    })


@app.route("/api/me")
def current_user():

    if "user_email" not in session:

        return jsonify({
            "logged_in": False
        })

    return jsonify({
        "logged_in": True,
        "name": session.get("user_name"),
        "email": session.get("user_email")
    })


# =========================================================
# HEALTH
# =========================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "Face Emotion Detection API"
    })


@app.route("/api/cv2-test")
def cv2_test():

    return jsonify({
        "cv2_file": str(getattr(cv2, "__file__", None)),
        "cv2_version": str(getattr(cv2, "__version__", None)),
        "cascade_classifier": hasattr(cv2, "CascadeClassifier"),
        "cascade_path": CASCADE_PATH,
        "cascade_exists": os.path.exists(CASCADE_PATH)
    })


# =========================================================
# EMOTION ANALYSIS
# =========================================================

@app.route("/api/analyze-face", methods=["POST"])
def analyze_face():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image provided."
        }), 400

    image_file = request.files["image"]

    if not image_file or not image_file.filename:

        return jsonify({
            "success": False,
            "error": "Empty image file."
        }), 400

    try:

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Image file is empty."
            }), 400

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "error": "Could not decode image."
            }), 400

        # -------------------------------------------------
        # Resize very large camera frames
        # -------------------------------------------------

        max_width = 960

        if frame.shape[1] > max_width:

            scale = max_width / frame.shape[1]

            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * scale),
                    int(frame.shape[0] * scale)
                )
            )

        # -------------------------------------------------
        # DeepFace
        # -------------------------------------------------

        print(
            f"Analyzing frame: {frame.shape}"
        )

        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        if isinstance(results, dict):
            results = [results]

        faces = []

        # -------------------------------------------------
        # Process detected faces
        # -------------------------------------------------

        for number, result in enumerate(
            results,
            start=1
        ):

            region = result.get("region") or {}

            emotions_raw = (
                result.get("emotion") or {}
            )

            x = int(region.get("x", 0) or 0)
            y = int(region.get("y", 0) or 0)
            w = int(region.get("w", 0) or 0)
            h = int(region.get("h", 0) or 0)

            if w <= 0 or h <= 0:
                continue

            emotions = {}

            for emotion_name, score in emotions_raw.items():

                try:

                    emotions[
                        str(emotion_name).lower()
                    ] = round(
                        float(score),
                        2
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    emotions[
                        str(emotion_name).lower()
                    ] = 0.0

            dominant = str(
                result.get(
                    "dominant_emotion",
                    "neutral"
                )
            ).lower()

            confidence = round(
                float(
                    emotions.get(
                        dominant,
                        0
                    )
                ),
                2
            )

            faces.append({
                "id": number,
                "emotion": dominant,
                "confidence": confidence,
                "emotions": emotions,
                "region": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                }
            })

        # -------------------------------------------------
        # No face
        # -------------------------------------------------

        if not faces:

            return jsonify({
                "success": False,
                "face_count": 0,
                "faces": [],
                "error": "No face detected."
            }), 200

        # -------------------------------------------------
        # Primary face
        # -------------------------------------------------

        primary = max(
            faces,
            key=lambda face: face["confidence"]
        )

        return jsonify({
            "success": True,
            "emotion": primary["emotion"],
            "confidence": primary["confidence"],
            "emotions": primary["emotions"],
            "face_count": len(faces),
            "faces": faces
        })

    except Exception as error:

        print("\n==============================")
        print("DEEPFACE ERROR")
        print("==============================")

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )

    print(
        f"EmotiVision running on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )