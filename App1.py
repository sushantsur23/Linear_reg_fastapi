from flask import Flask, request, render_template,redirect,url_for
from flask_cors import cross_origin
import sklearn
import pickle
#from Logger import Logger
import pandas as pd
import csv

app = Flask(__name__)

model = pickle.load(open("linear_reg.pkl", "rb"))
@app.route("/",methods = ["GET","POST"])

def predict():

    try:
        Pressure_Temp = request.form["PT"]
        Rotation_speed = request.form["RS"]
        Torque = request.form["Torque"]
        Tool_wear = request.form["TW"]
        TWF =  request.form["TWF"]
        HDF = request.form["HDF"]
        PWF = request.form["PWF"]
        OSF = request.form["OSF"]
        RNF = request.form["RNF"]


        prediction = model.predict([[
            Pressure_Temp,
            Rotation_speed,
            Torque,
            Tool_wear,
            TWF,
            HDF,
            PWF,
            OSF,
            RNF
        ]])

        output =round(prediction[0],2)
        print(output)
#        return render_template('home.html',Atmosphere_pressure="The atmosphere pressure is {} ".format(
#                                   output))
    except Exception as a:
        print('Operation not successful' + str(a))

if __name__ == "__main__":
    app.run(debug = True)