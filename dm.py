import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
# %matplotlib inline
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize']=20,10
#from keras.models import Sequential
#from keras.layers import LSTM,Dropout,Dense
#from sklearn.preprocessing import MinMaxScaler
#import xlrd

import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Read in data and display first 5 rows
def buildModel(compareArray):
    features = pd.read_csv('healthcost.csv')
    features.head(5)


    # Use numpy to convert to arrays
    # Labels are the values we want to predict
    labels = np.array(features['TOTEXP19'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features= features.drop('TOTEXP19', axis = 1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
 
    features = np.array(features)
    #print(features)

    # Using Skicit-learn to split data into training and testing sets
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    #print('Training Features Shape:', train_features.shape)
    #print('Training Labels Shape:', train_labels.shape)
    #print('Testing Features Shape:', test_features.shape)
    #print('Testing Labels Shape:', test_labels.shape)

    # Import the model we are using
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)


    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')
    finalPrediction = rf.predict(compareArray)
    return finalPrediction

    #print(test_features)
    #print(predictions)


