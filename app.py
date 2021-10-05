import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Trees_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        orderOnline = int(request.form['Order Online'])
        bookTable = int(request.form['Book Table'])
        votes = int(request.form['Votes'])
        location = int(request.form['Location'])
        restaurantType = int(request.form['Restaurant Type'])
        cuisiens = int(request.form['Cuisiens'])
        cost = int(request.form['Cost'])
        menuItems = int(request.form['Menu Items'])

        prediction = model.predict([[orderOnline,bookTable,votes,location,restaurantType,cuisiens,cost,menuItems]])
        output = round(prediction[0],1)
        return render_template('index.html',prediction_text='Thank you! Your Restaurant rating is {}'.format(output))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
