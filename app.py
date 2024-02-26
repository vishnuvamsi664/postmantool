from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application
app.config['DEBUG'] = True

# Route for a home page
@app.route('/')
def index():
    return "Welcome to the Home Page"

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = request.get_json()  # Assuming input is in JSON format
    
        pred_df = pd.DataFrame([data])

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Append the prediction to the input data
        output = {**data, 'prediction': float(results[0])}

        # Return the combined input and prediction as JSON
        return jsonify(output)
    else:
        return "Invalid Request Method"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)