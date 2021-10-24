 #import PySimpleGUI as sg
 #sg.Window(title="Jake has the nastiest chode i've ever seen",layout =[[]],margins=(100,50)).read()

 # hello_world.py

from tkinter.constants import TRUE
import PySimpleGUI as sg
from dm import buildModel
import numpy as np
import csv
import pandas as pd

def main():
    sg.theme("DarkBlue15")

    layout = [[sg.Text("Tell us about you")], 
    [sg.Text("Height: "), sg.Input(key="feet" ,change_submits=True, size=(2,40)), sg.Text("feet"), sg.Input(key="inches" ,change_submits=True, size=(4,40)), sg.Text("inches")],
    [sg.Text("Weight: "), sg.Input(key="wt" ,change_submits=True, size=(5,40)), sg.Text("lbs")],
    [sg.Text("Age: "), sg.Input(key="age" ,change_submits=True, size=(3,40)), sg.Text("years")],
    [sg.T("")],
    [sg.Text("Conditions: ")],
    [sg.Checkbox('High Cholesterol', default=False, key="High Cholesterol")], 
    [sg.Checkbox('High Blood Pressure', default=False, key="High Blood Pressure")],
    [sg.Checkbox('Diabetes', default=False, key="Diabetes")], 
    [sg.Checkbox('Cancer', default=False, key="cancer")],
    #[sg.Checkbox('Heart Disease', default=False, key="heart disease" )],
    [sg.Checkbox('Do you smoke', default=False, key="smoke")],
    [sg.Text("How Would you describe your overall health"), sg.Radio('Poor', "RADIO1", key="poorHealth"), sg.Radio('Fair', "RADIO1", key="fairHealth"), sg.Radio('Good', "RADIO1", key="goodHealth"),sg.Radio('Very Good', "RADIO1", key="veryGoodHealth"), sg.Radio('Excellent', "RADIO1", key="excellentHealth")],
    #[sg.Text("What is your annual income"), sg.Radio('less than 30k', "RADIO2"), sg.Radio('30,001-61,596', "RADIO2"), sg.Radio('61,597-110,787', "RADIO2"),sg.Radio('greater than $110,787', "RADIO2")],
    [sg.Text("What is your annual income"),  sg.Input(key="income" ,change_submits=True, size=(7,40)), sg.Text("dollars")],
    [sg.Text("I engage in activities (work, sports, etc.) that put me at risk for accidental injuries."), sg.Radio('Disagre Strongly', "RADIO3", key="lowDanger"), sg.Radio('Disagree Somewhat', "RADIO3", key="midLowDanger"), sg.Radio('Uncertain', "RADIO3", key="midDanger"),sg.Radio('Agree Somewhat', "RADIO3", key="midHighDanger"), sg.Radio('Agree Strongly',"RADIO3", key="highDanger")],
    [sg.Button("Submit",size=(20,4))]]

    window = sg.Window("intake form", layout, margins=(200,150))

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "Submit" or event == sg.WIN_CLOSED:
            if values["age"] != "":
                age19x = int(values["age"])

            if values["High Cholesterol"] == True:
                choldx = 1
            else:
                choldx=2
            
            if values["High Blood Pressure"] == True:
                hibpdx = 1
            else:
                hibpdx=2
            
            if values["Diabetes"] == True:
                diabdx_M18 = 1
            else:
                diabdx_M18=2

            if values["cancer"] == True:
                cancerdx = 1
            else:
                cancerdx = 2

            if values["smoke"] == True:
                adsmok42 = 1
            else:
                adsmok42 = 2


            if values["poorHealth"] == True:
                adgenh42=5
            elif values["fairHealth"]==True:
                adgenh42=4
            elif values["goodHealth"]==True:
                adgenh42=3
            elif values["veryGoodHealth"]==True:
                adgenh42=2
            elif values["excellentHealth"]==True:
                adgenh42=1

            if values["income"] != "":
                faminc19 = int(values["income"])

            if values["lowDanger"] == True:
                adrisk42=1
            elif values["midLowDanger"]==True:
                adrisk42=2
            elif values["midDanger"]==True:
                adrisk42=3
            elif values["midHighDanger"]==True:
                adrisk42=4
            elif values["highDanger"]==True:
                adrisk42=5

            #if values["feet"] != "" and values["inches"] != "" and values["wt"] != "":
                #BMInote = str(getBMI(values["feet"],values["inches"],values["wt"]))
                #print("your BMI is " + BMInote)
            


            participant = [cancerdx, choldx, diabdx_M18, hibpdx, age19x, adgenh42, adsmok42, faminc19, adrisk42]
            labels = ["cancerdx", "choldx", "diabdx_M18", "hibpdx", "age19x", "adgenh42", "adsmok42", "faminc19", "adrisk42"]

            f = open('participantList.csv', 'w')
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow(labels)
            writer.writerow(participant)
            f.close()

            newCost = CompareNewParticipant('ParticipantList.csv')


            #print(participant)
            break
    window.close()
    
    bestPlan = CompareCosts(newCost)
    displyWinner(bestPlan)


def getBMI(ft, inches, weight):
    height=int(ft)*12+int(inches)
    bmi=0
    if height != 0:
        bmi=int(weight)/height/height*703
    return bmi 

def CompareNewParticipant(participantListCSV):
    data = pd.read_csv(participantListCSV)
    print(data)
    lastRow = data.tail(1)
    compare = np.array(lastRow)
    #print(compare)
    answer = buildModel(compare)
    #print("the answer is ", answer)
    return int(answer)

def CompareCosts(cost):
    goldPrice = 700+ 78*12
    
    if cost*.10 +500 > 2000:
        advantagePrice= 2000 + 70*12
    else:
        advantagePrice = cost*.10 + 500 + 70*12
    
    if cost*.20 + 750 > 3000:
        plusPrice = 3000 + 70*12
    else:
        plusPrice = cost*.20 +750 + 24*12
    
    if cost*.30 + 1500 > 5000:
        basicPrice = 5000 + 0
    else:
        basicPrice = cost*.30

    lowest = min([goldPrice, advantagePrice, plusPrice, basicPrice])
    if goldPrice==lowest:
        return "Panther Gold"
    elif advantagePrice==lowest:
        return "Panther Advantage"
    elif plusPrice == lowest:
        return "Panther Plus"
    else:
        return "Panther Basic"

def displyWinner(plan):
    sg.theme("DarkBlue6")

    layout = [[sg.Text("Based on your information you would save the most money be retaining the most care by opting for:")], 
    [sg.Text(plan, size=(30,1), font=("Helvetica",25))],
    [sg.Button("Thank You", size=(20,4))]]

    window = sg.Window("Best Choice", layout, margins=(100,75))
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "Thank You" or event == sg.WIN_CLOSED:
            break
        window.close()






if __name__ == '__main__': #This is our call to the main function.
	main()




"""
class datamodel:
    def buildModel():
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
        print(features)
        features = np.array(features)


        # Using Skicit-learn to split data into training and testing sets
        from sklearn.model_selection import train_test_split
        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # Import the model we are using
        from sklearn.ensemble import RandomForestRegressor
        # Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        # Train the model on training data
        rf.fit(train_features, train_labels)


        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)
        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')
        print(test_features)
        print(predictions)
"""