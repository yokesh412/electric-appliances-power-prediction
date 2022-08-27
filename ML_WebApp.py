# Import libraries
import numpy as np


from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('/content/drive/MyDrive/model_pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)    
