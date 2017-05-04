# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:52:36 2017

@author: Sonali Majumdar
"""

import csv, pylab
import numpy as np

def openDataFile(datafile):
    '''
    Opens a csv file for reading  
    reads the required columns
    the required columns are 'Daily global solar exposure (MJ/m*m)',
    'Maximum temperature (Degree C)' and 'Rainfall amount (millimetres)'
    returns the column values 
    '''
    
    with open(datafile) as csvfile:
        reader = csv.DictReader(csvfile)
        solar_exposure = []
        max_temperature = []
        rainfall_amount = []
        for row in reader:
            solar_exposure.append(row['Daily global solar exposure (MJ/m*m)'])
            max_temperature.append(row['Maximum temperature (Degree C)'])
            rainfall_amount.append(row['Rainfall amount (millimetres)'])

        return (solar_exposure, max_temperature, rainfall_amount)

def computeCoefficients(values):
    '''
    given multiple variables, 
    computes the linear regression coefficients
    returns the generated coefficients
    '''
    x = np.array([values[1], values[2]])
    X = np.vstack([np.ones(np.max(x.shape)), x]).T
    y = np.array(values[0])
    a = np.linalg.lstsq(X, y)[0]
    return a

def predictYGivenCoefficients(coefficients, values):
    '''
    y = B0 + b1x1 + b2x2 
    predicts a y value based on given x values and coefficients
    '''
    predict_y = coefficients[0] + coefficients[1] * np.array(values[1], dtype=np.float) + coefficients[2] * np.array(values[2], dtype=np.float)
    print(predict_y)
    return predict_y;
    
    
def plotGraph(x,y):
    '''
    plots the values for x and y
    uses pylab for plotting
    '''      
    pylab.plot(x, y, '-r', label='By temperature')
    pylab.xlabel('Maximum Temperature Forecast')
    pylab.ylabel('Predicted Solar exposure')
    pylab.ylim(11,15)
    pylab.title('Predicting solar exposure')
    pylab.show()
    
values = openDataFile('samplecsv.csv')
coefficients = computeCoefficients(values)
print (coefficients)
max_temperature_forecast = [16.7, 18, 18, 15, 16, 17, 18, 18]
rainfall_amount_forecast = [0, 0, 0, 0, 0, 0.1, 0.1, 0.1]
estimated_y = predictYGivenCoefficients(coefficients, ([12.3], max_temperature_forecast, rainfall_amount_forecast))
plotGraph(max_temperature_forecast, estimated_y)

