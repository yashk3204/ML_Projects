from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Loading the models as .pkl files
with open('naive_bayes.pkl', 'rb') as file1:
    nb_model = pickle.load(file1)
with open('perceptron.pkl', 'rb') as file2:
    perceptron_model = pickle.load(file2)

# Making the predict API method
@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'naive_bayes')
    input_features = np.array([[ data['age'], data['glucose'], data['insulin'], data['bmi'] ]])
    if model_type == 'naive_bayes':
        prediction = nb_model.predict(input_features)
    elif model_type == 'perceptron':
        prediction = perceptron_model.predict(input_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    return jsonify({'diabetes_type': int(prediction[0])})

# Running the app
if __name__ == '__main__':
    app.run(debug=True)