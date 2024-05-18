from flask import Flask, request, jsonify
from sklearn.linear_model import Perceptron
import numpy as np

# Create a flask
app = Flask(__name__)

X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 1, 0])

model = Perceptron()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['features']])
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run() # domyślnie ustawia localhost i port 5000
    # app.run(host='0.0.0.0', port=8000)
