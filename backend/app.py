
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Train ML model
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([35,40,45,50,55,60,65,70,75,80])

model = LinearRegression()
model.fit(X, y)

@app.route("/")
def home():
    return "Student Marks Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hours = float(data["hours"])
    prediction = model.predict([[hours]])
    return jsonify({"predicted_marks": round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
