from flask import Flask, request, jsonify
import joblib
import sqlite3
from urllib.parse import parse_qs

app = Flask(__name__)
model = joblib.load("dengue_model.pkl")

def create_database():
    conn = sqlite3.connect("dengue.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            fever REAL,
            headache REAL,
            joint_pain REAL,
            vomiting REAL,
            rash REAL,
            eye_pain REAL,
            fatigue REAL,
            muscle_pain REAL,
            nausea REAL,
            prediction TEXT,
            risk_level TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_patient(data, prediction, risk_level):
    conn = sqlite3.connect("dengue.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (
            name, age, gender, fever, headache, joint_pain,
            vomiting, rash, eye_pain, fatigue, muscle_pain,
            nausea, prediction, risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("Name", ""),
        int(float(data.get("Age", 0))),
        data.get("Gender", ""),
        float(data.get("Fever", 0)),
        float(data.get("Headache", 0)),
        float(data.get("JointPain", 0)),
        float(data.get("Vomiting", 0)),
        float(data.get("Rash", 0)),
        float(data.get("EyePain", 0)),
        float(data.get("Fatigue", 0)),
        float(data.get("MusclePain", 0)),
        float(data.get("Nausea", 0)),
        prediction,
        risk_level
    ))
    conn.commit()
    conn.close()

create_database()

@app.route("/")
def home():
    return "Dengue Prediction API is Running"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            data = request.args.to_dict()
        else:
            raw_data = request.get_data(as_text=True)
            if raw_data:
                parsed = parse_qs(raw_data)
                data = {key: value[0] for key, value in parsed.items()}
            else:
                data = request.form.to_dict()

        print("RECEIVED DATA:", data)

        fever = float(data.get("Fever", 0))
        headache = float(data.get("Headache", 0))
        joint_pain = float(data.get("JointPain", 0))
        vomiting = float(data.get("Vomiting", 0))
        rash = float(data.get("Rash", 0))
        eye_pain = float(data.get("EyePain", 0))
        fatigue = float(data.get("Fatigue", 0))
        muscle_pain = float(data.get("MusclePain", 0))
        nausea = float(data.get("Nausea", 0))
        fever_days = float(data.get("FeverDays", 0))

        features = [[
    fever,
    headache,
    joint_pain,
    vomiting,
    rash
]]

        result = model.predict(features)[0]
        result_text = str(result).strip().lower()

        if result_text in ["1", "yes", "true", "dengue", "dengue detected"]:
            prediction = "Dengue Detected"
        else:
            prediction = "No Dengue Detected"

        average_score = (
            fever + headache + joint_pain + vomiting +
            rash + eye_pain + fatigue + muscle_pain + nausea
        ) / 9

        if average_score >= 7:
            risk_level = "HIGH"
            message = "High predicted risk. Please consult a healthcare professional."
        elif average_score >= 4:
            risk_level = "MODERATE"
            message = "Moderate predicted risk. Please consult a healthcare professional."
        else:
            risk_level = "LOW"
            message = "Low predicted risk. If symptoms continue or become worse, please consult a healthcare professional."

        save_patient(data, prediction, risk_level)

        return jsonify({
            "prediction": prediction,
            "risk_level": risk_level,
            "message": message
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "prediction": "Error",
            "risk_level": "UNKNOWN",
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

