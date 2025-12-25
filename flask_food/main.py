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


# @app.route("/api/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     x_input = pd.DataFrame([data], columns=colums)
#     x_input_scaled = loaded_scaler.transform(x_input)

#     # Debug: lihat nilai setelah scaling
#     print("Input original:", x_input)
#     print("Input scaled:", x_input_scaled)

#     prediction = loaded_model.predict(x_input_scaled)
#     predicted_value = prediction.tolist()[0]
#     print("Prediction value:", predicted_value)

#     # Jika hasil prediksi > 500, kategori "Sehat", selainnya "Tidak sehat"
#     # Sesuaikan threshold 500 dengan data training Anda
#     health_status = "Sehat" if predicted_value > 500 else "Tidak sehat"
#     return jsonify(
#         {
#             "meta": {"status": "success", "message": "Prediction made successfully"},
#             "data": health_status,
#         }
#     )


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint untuk menghitung nilai nutrisi makanan
    
    Input JSON:
    {
        "food_name": "nama makanan",
        "category": "kategori",
        "calories": nilai_kalori,
        "protein": nilai_protein,
        "carbs": nilai_karbohidrat,
        "fat": nilai_lemak,
        "iron": nilai_zat_besi,
        "vitamin_c": nilai_vitamin_c
    }
    """
    try:
        data = request.get_json()
        
        # Validasi input
        required_fields = colums
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "meta": {"status": "error", "message": f"Field '{field}' diperlukan"},
                    "data": None
                }), 400
        
        # Ekstrak data
        food_name = data.get("food_name")
        category = data.get("category")
        calories = float(data.get("calories", 0))
        protein = float(data.get("protein", 0))
        carbs = float(data.get("carbs", 0))
        fat = float(data.get("fat", 0))
        iron = float(data.get("iron", 0))
        vitamin_c = float(data.get("vitamin_c", 0))
        
        # Perhitungan nilai gizi
        total_macronutrients = protein + carbs + fat
        protein_percentage = (protein / total_macronutrients * 100) if total_macronutrients > 0 else 0
        carbs_percentage = (carbs / total_macronutrients * 100) if total_macronutrients > 0 else 0
        fat_percentage = (fat / total_macronutrients * 100) if total_macronutrients > 0 else 0
        
        # Nilai energi dari macronutrient (4 kal/g protein, 4 kal/g carbs, 9 kal/g fat)
        energy_from_macros = (protein * 4) + (carbs * 4) + (fat * 9)
        
        # Perhitungan skor kesehatan (0-100)
        health_score = min(100, max(0, 
            (protein * 0.25) + (iron * 2) + (vitamin_c * 0.5) - (fat * 0.1)
        ))
        
        # Kategori kesehatan detail
        if health_score >= 80:
            nutrition_status = "Sangat Sehat"
        elif health_score >= 60:
            nutrition_status = "Sehat"
        elif health_score >= 40:
            nutrition_status = "Cukup"
        else:
            nutrition_status = "Kurang Sehat"
        
        # Kategori kesehatan simple (Sehat / Tidak Sehat)
        # Makanan dianggap Sehat jika health_score >= 60
        is_healthy = health_score >= 60
        health_category = "Sehat" if is_healthy else "Tidak Sehat"
        
        # Return hasil perhitungan
        return jsonify({
            "meta": {
                "status": "success",
                "message": "Perhitungan nutrisi berhasil"
            },
            "data": {
                "food_info": {
                    "name": food_name,
                    "category": category
                },
                "nutrient_values": {
                    "calories": round(calories, 2),
                    "protein": round(protein, 2),
                    "carbs": round(carbs, 2),
                    "fat": round(fat, 2),
                    "iron": round(iron, 2),
                    "vitamin_c": round(vitamin_c, 2)
                },
                "macronutrient_breakdown": {
                    "total_macronutrients": round(total_macronutrients, 2),
                    "protein_percentage": round(protein_percentage, 2),
                    "carbs_percentage": round(carbs_percentage, 2),
                    "fat_percentage": round(fat_percentage, 2)
                },
                "energy_calculation": {
                    "from_protein": round(protein * 4, 2),
                    "from_carbs": round(carbs * 4, 2),
                    "from_fat": round(fat * 9, 2),
                    "total_energy_calculated": round(energy_from_macros, 2)
                },
                "health_assessment": {
                    "health_score": round(health_score, 2),
                    "detailed_status": nutrition_status,
                    "is_healthy": is_healthy,
                    "health_category": health_category
                }
            }
        })
        
    except ValueError as e:
        return jsonify({
            "meta": {"status": "error", "message": "Nilai numerik tidak valid"},
            "data": None
        }), 400
    except Exception as e:
        return jsonify({
            "meta": {"status": "error", "message": str(e)},
            "data": None
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
