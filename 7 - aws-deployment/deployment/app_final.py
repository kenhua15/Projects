import flask
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, Markup

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on housing prices,
# Build a LinearRegression predictor on it
housing = pd.read_csv("train.csv")

X = housing[['FullBath', 'BedroomAbvGr', 'GrLivArea']]
Y = housing['SalePrice']
PREDICTOR = LinearRegression().fit(X, Y)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)


# Homepage


# Get an example and return it's score from the predictor model

@ app.route('/')
def crop_recommend():
    title = 'House Price Predictor'
    return render_template('crop.html', title=title)


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'House Price Predictor'

    if request.method == 'POST':
        Bath = int(request.form['bath'])
        Bed = int(request.form['bed'])
        Area = int(request.form['area'])
        data = np.array([[Bath, Bed, Area]])
        my_prediction = PREDICTOR.predict(data)
        final_prediction = my_prediction[0]



        return render_template('crop-result.html', prediction=final_prediction, title=title)


#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port = 8080)
app.run(debug=True)
