import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Trees_model.pkl','rb'))

@app.route('/home',methods=['GET'])
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],1)
    return render_template('index2.html',prediction_text='Thank you! Your Restaurant rating is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
