# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.



## THEORY

The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains nine neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model.
Data is the key for the working of neural network and we need to process it before feeding to the neural network. In the first step, we will visualize data which will help us to gain insight into the data.We need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size.Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate.Fitting is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network.After fitting model, we can test it on test data to check whether the case of overfitting. 

## Neural Network Model

![snip1](https://user-images.githubusercontent.com/75235209/187214846-d5a573b6-2952-44d3-8f2d-84b48a85655e.PNG)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
#Developed by R.Ramanesh
# Register number: 212220040130

iimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv("dataset1.csv")
data.head()
x=data[['Input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI_BRAIN=Sequential([
    Dense(9,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1)
])
AI_BRAIN.compile(optimizer='rmsprop', loss='mse')
AI_BRAIN.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(AI_BRAIN.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
AI_BRAIN.evaluate(x_test1,y_test)
x_n1=[[26]]
x_n1_1=Scaler.transform(x_n1)
AI_BRAIN.predict(x_n1_1)


```
## Dataset Information
![image](https://user-images.githubusercontent.com/75235209/187224852-6acca557-84e0-44ba-9c81-07ab4175d19b.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![loss](https://user-images.githubusercontent.com/75235209/187225297-db11357c-2e40-459e-af90-187fc068a336.PNG)


### New Sample Data Prediction
![output](https://user-images.githubusercontent.com/75235209/187225587-e7cbc048-ad4b-4541-b99e-ebf5c0f49a2e.PNG)


## RESULT

Thus,the neural network regression model for the given dataset is developed.
