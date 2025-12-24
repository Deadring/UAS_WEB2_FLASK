from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib


app = Flask(__name__)
CORS(app)

loaded_model = joblib.load("models/food_nutrition_model.pkl")

loaded_scaler = joblib.load("models/food_nutrition_preprocessor.pkl")

colums = [
    "food_name",
    "category",
    "calories",
    "protein",
    "carbs",
    "fat",
    "iron",
    "vitamin_c",
]


@app.route("/")
def index():
    return jsonify(
        {
            "meta": {
                "status": "success",
                "message": "Welcome to the Food Nutrition Prediction API",
            },
            "data": None,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x_input = pd.DataFrame([data], columns=colums)
    x_input_scaled = loaded_scaler.transform(x_input)

    # Debug: lihat nilai setelah scaling
    print("Input original:", x_input)
    print("Input scaled:", x_input_scaled)

    prediction = loaded_model.predict(x_input_scaled)
    predicted_value = prediction.tolist()[0]
    print("Prediction value:", predicted_value)

    # Jika hasil prediksi > 500, kategori "Sehat", selainnya "Tidak sehat"
    # Sesuaikan threshold 500 dengan data training Anda
    health_status = "Sehat" if predicted_value > 500 else "Tidak sehat"
    return jsonify(
        {
            "meta": {"status": "success", "message": "Prediction made successfully"},
            "data": health_status,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
