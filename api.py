from flask import Flask, request, jsonify
import joblib
import sqlite3

app = Flask(__name__)

# Load trained model
model = joblib.load("dengue_model.pkl")


@app.route("/")
def home():
    return "Dengue Prediction API is Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data from MIT App Inventor
        data = request.get_json(force=True)

        # Read user details
        name = data.get("Name", "")
        age = data.get("Age", "")
        gender = data.get("Gender", "")

        # Read symptoms
        fever = int(data.get("Fever", 0))
        headache = int(data.get("Headache", 0))
        jointpain = int(data.get("JointPain", 0))
        vomiting = int(data.get("Vomiting", 0))
        rash = int(data.get("Rash", 0))

        # Predict
        result = model.predict([[fever, headache, jointpain, vomiting, rash]])

        if result[0] == "Yes":
            prediction = "Dengue Detected"
            risk = "High"
        else:
            prediction = "No Dengue"
            risk = "Low"

        # Save to database
        conn = sqlite3.connect("patients.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO patients
        (name, age, gender, fever, headache, jointpain, vomiting, rash, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            age,
            gender,
            fever,
            headache,
            jointpain,
            vomiting,
            rash,
            prediction
        ))

        conn.commit()
        conn.close()

        # Return JSON response
        return jsonify({
            "prediction": prediction,
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)