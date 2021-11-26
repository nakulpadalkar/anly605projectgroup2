from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from sklearn.linear_model import Ridge, Lasso, LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import uuid 
import os
import plotly.graph_objects as go
import plotly
app = Flask(__name__)

# @app.route("/") # Start here
@app.route("/",methods=['GET','POST']) # We need to change the first line to include GET and POST methods


def hello_world():
    request_type_str = request.method
    if request_type_str=='GET':
        return render_template("index.html",href="static/attempt1.svg")
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/"+random_string +".svg"

        # Load and Create Dataframe
        heart_group2 = pd.read_csv('heart_failure_clinical_records_dataset.csv', header = 0)
        x = heart_group2.iloc[:,:12]
        y = heart_group2.iloc[:,12]

        # Split the data frame
        X_train_group2, X_test_group2, y_train_group2, y_test_group2 = train_test_split(x, y, test_size=0.1, random_state=8281)
        

        np_arr = floatsome_to_np_array(text)
        pkl_filename="TrainedModel/pickle.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
            model = list(pickle_model.values())[0] 
            model.fit(X_train_group2,y_train_group2)

        plot_graphs(model=model,new_input_arr=np_arr,output_file= path)
        return render_template("index.html",href=path)


def plot_graphs(model,new_input_arr):
    heart_group2 = pd.read_csv('heart_failure_clinical_records_dataset.csv', header = 0)
    fig = plotly.subplots.make_subplots(
    rows=1, cols=2
    )

    fig.add_trace(
        go.Scatter(x=heart_group2["age"],y=heart_group2['time'],mode='markers',
        marker=dict(
                color=heart_group2['DEATH_EVENT']),
            line=dict(color="#003366",width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=heart_group2['ejection_fraction'],y=heart_group2['serum_sodium'],mode='markers',
        marker=dict(
                color=heart_group2['DEATH_EVENT']),
            line=dict(color="#FF6600",width=1)),
        row=1, col=2
    )

    new_preds = model.predict(new_input_arr)
    Age_input = np.array(new_input_arr[0][0])
    Ejection_input =np.array(new_input_arr[0][4])
    Time_input = np.array(new_input_arr[0][11])
    Serum_input =np.array(new_input_arr[0][8])

    fig.add_trace(
    go.Scatter(
        x=Age_input,
        y=Ejection_input,
        mode='markers', name="Scatter Plot1",
        marker=dict(
            color=heart_group2['DEATH_EVENT'],size=15),
        line=dict(color="#FFCC00",width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=Time_input,
            y=Serum_input,
            mode='markers', name="Scatter Plot2",
            marker=dict(
                color=heart_group2['DEATH_EVENT'],size=15),
            line=dict(color="red",width=1)),
            row=1, col=2
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Ejection Fraction", row=1, col=1)
    fig.update_yaxes(title_text="Serum Sodium", row=1, col=2)
    # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    # Update title and height
    fig.update_layout(height=400, width=800, title_text="Variation in Heart Disease Death Rate")
    output_file="static/final.svg"
    fig.write_image(output_file,width=1200,engine="kaleido")
    fig.show()


def floatsome_to_np_array(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(1,len(floats))






# example comment made by me














# def hello_world():
#     request_type_str = request.method
#     if request_type_str == 'GET':
#         return render_template('index.html', href='static/base_pic.svg')
#     else:
#         text = request.form['text']
        # random_string = uuid.uuid4().hex
        # path = "static/" + random_string + ".svg"
        # model = load('model.joblib')
        # np_arr = floats_string_to_np_arr(text)
        # make_picture('AgesAndHeights.pkl', model, np_arr, path)
    # return render_template("index.html",href=path)

# This is the first and second step
# def hello_world():
#     return "<p>Hello, World</p>" # First Step
#     # return render_template('index.html', href='static/baseimage.svg') # Comment first step and then uncomment this

# def hello_world():
#     request_type_str = request.method
#     if request_type_str == 'GET':
#         return render_template('index.html', href='static/base_pic.svg')
#     else:
#         text = request.form['text']
        # random_string = uuid.uuid4().hex
        # path = "static/" + random_string + ".svg"
        # model = load('model.joblib')
        # np_arr = floats_string_to_np_arr(text)
        # make_picture('AgesAndHeights.pkl', model, np_arr, path)
    # return render_template("index.html",href=path)

    # boston = load_boston()
    # pkl_filename = "TrainedModel/StackedPickle.pkl"
    # testvalue = boston.data[1].reshape(1, -1)
    # test_input = testvalue
    # with open(pkl_filename, 'rb') as file:
    #     pickle_model = pickle.load(file)
    # predict = pickle_model.predict(test_input)
    # predict_as_str = str(predict)
    # return predict_as_str