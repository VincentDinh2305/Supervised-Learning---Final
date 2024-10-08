from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        trsn_city_veh = float(request.form['trsn_city_veh'])
        pedestrian = float(request.form['pedestrian'])
        traffctl = float(request.form['traffctl'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        year = int(request.form['year'])
        motorcycle = float(request.form['motorcycle'])
        automobile = float(request.form['automobile'])
        cyclist = float(request.form['cyclist'])
        truck = float(request.form['truck'])
        model = request.form['model']

        # Get one directory up from current directory
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.dirname(path)

        # Load the model
        loaded_model = joblib.load(path + '\\' + model + '_best_estimator.pkl')

        test_data = pd.DataFrame([[trsn_city_veh, traffctl, latitude, longitude, motorcycle, year, automobile, cyclist, truck, pedestrian]], columns=['trsn_city_veh', 'traffctl', 'latitude', 'longitude', 'motorcycle', 'year', 'automobile', 'cyclist', 'truck', 'pedestrian'])
        test_data.columns = test_data.columns.str.upper()

        print(test_data.head())
        print(test_data.info())

        # Make prediction
        prediction = loaded_model.predict(test_data)

        # Get the first and only value of the prediction
        prediction = prediction[0]

        if prediction == 0:
            prediction = 'Non-Fatal Accident'
        elif prediction == 1:
            prediction = 'Fatal Accident'

        # Show the prediction result in prediction.html
        return render_template('prediction.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
