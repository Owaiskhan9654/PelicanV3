# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request, send_from_directory
import utils
import os
import pandas as pd

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
# ‘/’ URL is bound with hello_world() function.
def login_landing_Page():
    error_login = None
    a = {'owais.ahmed@canarydetect.com': 'Canary@2021', 'r.varsha@canarydetect.com': 'Canary@2021'}

    if request.method == 'POST':
        if request.form['username'] in a and a[request.form['username']] == request.form['password']:
            all_files = [i for i in range(1, 6419)]
            sample_id_list = sorted(all_files, reverse=True)

            return render_template('index.html', show_results="false", sample_id_len=len(sample_id_list),
                                   sample_id_list=sample_id_list, len2=len([]),
                                   all_prediction_data=[],
                                   prediction_date="", dates=[], all_data=[], len=len([]))

        else:
            error_login = 'Invalid Credentials. Please try again.'

    return render_template('login.html', error_login=error_login)


@app.route('/process', methods=['POST'])
def process():
    all_prediction_data=[]
    all_files = [i for i in range(1, 6419)]
    sample_id_list = sorted(all_files,reverse=True)
    sample_id = request.form['SampleID']
    ml_algoritms = request.form.getlist('mlalgos')

    df = utils.get_data_from_database(sample_id)
    df, error_code = utils.data_check(df)
    #print(df)

    all_colors = {'linearmodel': '#FF9EDD',
                  'KNeighborsClassifier': '#FFFD7F',
                  'LinearSVC': '#FFA646',
                  'KernelSVC': '#CC2A1E',
                  'DecisionTreeClassifier': '#8F0099',
                  'RandomForestClassifier': '#CCAB43'}
    all_models = {'linearmodel': 'Logistic Regression with Grid Search',
                  'KNeighborsClassifier': 'K Nearest Neighbors Classifier with Grid Search',
                  'LinearSVC': 'Linear SVC (Support Vector Classifier) with Grid Search',
                  'KernelSVC': 'Kernel SVC (Support Vector Classifier) with Grid Search',
                  'DecisionTreeClassifier': 'Decision Tree Classifier with Grid Search',
                  'RandomForestClassifier': 'Random Forest Classifier with Grid Search'}
    if error_code is None:
        Predictions = []
        Prediction_probabilities = []
        features_df = utils.feature_extraction(df, sample_id)

        for i in ml_algoritms:

            if i == 'linearmodel':
                Prediction1, Prediction2 = utils.linear_model_predict(features_df)

            elif i == 'KNeighborsClassifier':
                Prediction1, Prediction2  = utils.KNeighborsClassifier_predict(features_df)

            elif i == 'LinearSVC':
                Prediction1, Prediction2 = utils.LinearSVC_predict(features_df)

            elif i == 'KernelSVC':
                Prediction1, Prediction2  = utils.kernalSVC_predict(features_df)

            elif i == 'DecisionTreeClassifier':
                Prediction1, Prediction2  = utils.DecisionTreeClassifier_predict(features_df)

            elif i == 'RandomForestClassifier':
                Prediction1, Prediction2  = utils.RandomForestClassifier_predict(features_df)
            model_name = all_models[i]
            Predictions.append({model_name:[{'Channel 1':Prediction1,'Channel 2':Prediction2}]})
        all_prediction_data.extend(Predictions)
    print(all_prediction_data)


    print(error_code)
    return render_template('index.html',all_test_evaluations=all_prediction_data, show_results="true", sample_id_len=len(sample_id_list), sample_id_list=sample_id_list,
                           len2=len(all_prediction_data),error_code=error_code,sample_id=sample_id,
                           all_prediction_data=all_prediction_data,Channels=['Channel 1','Channel 2'])

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True,host="127.0.0.2")
