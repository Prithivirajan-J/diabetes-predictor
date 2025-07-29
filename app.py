# app.py

from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        inputs = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # Scale input
        scaled_input = scaler.transform([inputs])

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1] * 100  # probability of having diabetes

        # Plot risk comparison
        avg_risk = 26.8  # Avg global diabetes prevalence %
        user_risk = round(proba, 2)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.bar(["Average", "You"], [avg_risk, user_risk], color=["gray", "crimson"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Diabetes Risk (%)")
        ax.set_title("Risk Comparison")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        plt.tight_layout()

        chart_path = "static/risk_chart.png"
        os.makedirs("static", exist_ok=True)
        plt.savefig(chart_path)
        plt.close()

        return render_template("result.html",
                               prediction=prediction,
                               probability=round(proba, 2),
                               chart_url=chart_path)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
