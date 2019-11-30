# Deep Learning: LSTM Cryptocurrency Price Predictor

*Does sentiment or closing prices provide a better signal for cryptocurrency price movements?* 

In this analysis, custom Long-short term memory (LSTM) models were used to predict Bitcoin's price. 

Long Short Term Memory model (LSTM) is a type of deep learning with a Recurrent Neural Network (RNN) architecture. It uses feedback connections to keep track of the dependencies between the elements in the input sequence and looks at the last “n” days(timestep) data to predict how the series can progress. 

For this analysis, sentiment and previous closing prices were used as inputs in the models to determine if they provide potential predictive patterns to forecast prices. For the sentiment data, the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) was used to buuld the first model. The second model used a lag of Bitcoin closing prices.  

Steps Completed: 

1. [Prepare the data for training and testing](Prepare-the-date-for-training-and-testing)
2. [Build and train a custom LSTM RNN](Build-and-train-a-custom-LSTM-RNN)
3. [Evaluate the performance of the model](Evaluate-the-performance-of-the-model)

## Preparing the data for training and testing 

Libraries `numpy` and `tensorflow` were imported. `Tensorflow` is a computational framework for building machine learning models. 

Once the data was imported, a function, `window_data` was applied to the data. This function created two new lists of the features data (X) and target data (y) grouped by the window, or lag of data the model would use to analyze. 
```
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number] #ndim = 2
        target = df.iloc[(i + window), target_col_number] #ndim = 1 
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1) #reshape array to ndim = 2
```
The data was split into 70% training and 30% test and scaled between 0 and 1 using `MinMaxScaler()` from the `sklearn` library.<br/>
<br/>
The freature data, X_train and X_test, were reshaped from two-dimensional to a three-dimensional array.<br/>
The LSTM network reads in three-dimensions for batch size, time-steps, and number of units in one input sequence. 

```
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))
```

## Evaluate the performance of the model 

> Which model has a lower loss?

> Which model tracks the actual values better over time?

> Which window size works best for the model?
