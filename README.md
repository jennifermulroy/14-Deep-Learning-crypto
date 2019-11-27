# Deep Learning: LSTM RNN Cryptocurrency Price Predictor

*Does sentiment or closing prices provide a better signal for cryptocurrency price movements?* 

In this analysis, custom Long-short term memory (LSTM) deep learning recurrent neural network (RNN) models were used to predict Bitcoin's closing price by incorporating sentiment and simple closing prices as inputs. 

The [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/), a sentiment measurement, uses a variety of data sources to produce a daily FNG value for cryptocurrency. The FNG values were used to build one model. The second model used a 10 day window of Bitcoin closing prices to predict the 11th day closing price.  

Steps Completed: 

1. [Prepare the data for training and testing](Prepare-the-date-for-training-and-testing)
2. [Build and train a custom LSTM RNN](Build-and-train-a-custom-LSTM-RNN)
3. [Evaluate the performance of the model](Evaluate-the-performance-of-the-model)

## Prepare the data for training and testing 

# Set the random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

1. Use the `window_data` function to generate the X and y values for the model.
2. Split the data into 70% training and 30% testing
3. Apply the MinMaxScaler to the X and y values
4. Reshape the X_train and X_test data for the model. Note: The required input format for the LSTM is:

## Evaluate the performance of the model 

> Which model has a lower loss?

> Which model tracks the actual values better over time?

> Which window size works best for the model?
