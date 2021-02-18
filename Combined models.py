# %% Importing Modules
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np
import math
import tensorflow as tf
from keras import backend as K
from keras  import callbacks
from keras import optimizers
from keras import initializers
from keras.models import Sequential,load_model
from keras.models import Model, load_model
from keras.engine.topology import Layer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, BatchNormalization, Reshape, Conv2D, MaxPool2D, Concatenate, Bidirectional, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
# from keras.layers import CuDNNGRU, CuDNNLSTM
from keras.optimizers import Adam

# %% Global Variables
Date_Column = 'Date'
Data_Column = 'Wind Speed'
Use_Percent = True
Testing_Percent = 0.20
Testing_Rows = 7
look_back = 10
epochs = 2
number_neural=1
future_pred_count=7
create_new_model = True
model_type = 'LSTMCapsule' # 'LSTM' 'BILSTM' 'GRU' 'LSTMCapsule' 'BILSTMCapsule' 'GRUCapsule'
model_name = 'LSTMCapsule.h5'
batch_size = 1

# %% Capsule
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule=1, dim_capsule=1, routings=3, kernel_size=(9, 1), share_weights=True, activation='default', initializer="he_normal", **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        self.initializer = initializers.get(initializer)

        self.units = 4
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)
        print("constructor")
    def __del__(self):
        print("destructor")

    def build(self, input_shape):
        self.w = self.add_weight( shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True, name="alpha" )
        self.b = self.add_weight( shape=(self.units,), initializer="random_normal", trainable=True, name="beta" )
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

        
    def call(self, inputs):
        # return tf.matmul(inputs, self.w) + self.b

        if self.share_weights:
            # u_hat_vecs = K.conv1d(inputs, self.W)
            u_hat_vecs = tf.matmul(inputs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(inputs, self.W, [1], [1])
        print("u_hat_vecs************************************")

        batch_size = K.shape(inputs)[0]
        print("batch_size************************************")

        input_num_capsule = K.shape(inputs)[1]
        print("input_num_capsule************************************")

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        print("u_hat_vecs************************************")

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        print("u_hat_vecs************************************")

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        print("b************************************")

        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            print("outputs************************************")
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])
        print("range************************************")
        return outputs

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Capsule, self).get_config()
        config = {"initializer": initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))

# %% 
def CreateModel():
    if model_type == 'LSTM':
        model = Sequential()
        model.add(LSTM(number_neural, batch_input_shape=(batch_size, look_back, 1), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    elif model_type == 'BILSTM':
        model = Sequential()
        # Input layer
        model.add(Bidirectional(LSTM(units = number_neural, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        # Hidden layer
        model.add(Bidirectional(LSTM(units = 4)))
        model.add(Dense(1))
        #Compile model
        model.compile(optimizer='adam',loss='mse')
        return model
    elif model_type == 'GRU':
        model = Sequential()
        # Input layer 
        model.add(GRU (units = number_neural, return_sequences = True, input_shape = [X_train.shape[1], X_train.shape[2]]))
        model.add(Dropout(0.2)) 
        # Hidden layer
        model.add(GRU(units = 4))                 
        model.add(Dropout(0.2))
        model.add(Dense(units = 1)) 
        #Compile model
        model.compile(optimizer='adam',loss='mse')
        return model
    if model_type == 'LSTMCapsule':
        model = Sequential()
        model.add(LSTM(number_neural, batch_input_shape=(batch_size, look_back, 1), stateful=True))
        model.add(Capsule(num_capsule=1, dim_capsule=1, routings=1, share_weights=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    elif model_type == 'BILSTMCapsule':
        model = Sequential()
        # Input layer
        model.add(Bidirectional(LSTM(units = number_neural, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        # Hidden layer
        model.add(Bidirectional(LSTM(units = 4)))
        model.add(Capsule(num_capsule=1, dim_capsule=1, routings=1, share_weights=True))
        model.add(Dense(1))
        #Compile model
        model.compile(optimizer='adam',loss='mse')
        return model
    elif model_type == 'GRUCapsule':
        model = Sequential()
        # Input layer 
        model.add(GRU (units = number_neural, return_sequences = True, input_shape = [X_train.shape[1], X_train.shape[2]]))
        model.add(Capsule(num_capsule=1, dim_capsule=1, routings=1, share_weights=True))
        model.add(Dropout(0.2)) 
        # Hidden layer
        model.add(GRU(units = 4))                 
        model.add(Dropout(0.2))
        model.add(Dense(units = 1)) 
        #Compile model
        model.compile(optimizer='adam',loss='mse')
        return model
# Plot train loss and validation loss
def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.savefig('loss_'+model_name+'.jpg', format='jpg', dpi=1000)


# %% Read Dataset
df = pd.read_csv('WindSpeed.csv', parse_dates=True, index_col=Date_Column)

# %% Plot Time Series
# Define a function to draw time_series plot (Day, Month, Year)
def timeseries (x_axis, y_axis, x_label):
    plt.figure(figsize = (10, 6))
    plt.plot(x_axis, y_axis, color ='black')
    plt.xlabel(x_label, {'fontsize': 12}) 
    plt.ylabel(Data_Column, {'fontsize': 12})
def plotdftimeseries(df):
    dataset = df.copy()
    timeseries(df.index, dataset[Data_Column], 'Time (day)')

    dataset['month'] = dataset.index.month
    dataset_by_month = dataset.resample('M').sum()
    timeseries(dataset_by_month.index, dataset_by_month[Data_Column], 'Time (month)')

    dataset['year'] = dataset.index.year
    dataset_by_year = dataset.resample('Y').sum()
    timeseries(dataset_by_year.index, dataset_by_year[Data_Column], 'Time (year)')
plotdftimeseries(df)

# %% Data Preprocessing
# Scaling between 0, 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df) 
print('Min', np.min(scaled))
print('Max', np.max(scaled))
print(scaled[:10])

# %% Split Training And Testing
if Use_Percent:
    train_size = int(len(df) * (1-Testing_Percent))
    test_size = len(df - train_size)
    pass
else:
    train_size = len(df) - Testing_Rows
    test_size = Testing_Rows
    pass
train, test = scaled[0:train_size, :], scaled[train_size: len(scaled), :]
print('train: {}\ntest: {}'.format(len(train), len(test)))

# %% Lookback
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
X_train, y_train = create_dataset(train, look_back)
X_valid, y_valid = X_train[-1000:], y_train[-1000:]
X_test, y_test = create_dataset(test, look_back)

# %% Adjusting Dimensions
# x shape : Rows , TimeSteps , Features
# y shape : Rows , 1
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape) 
print('y_test.shape: ', y_test.shape)


# %% Model
if create_new_model:
    model = CreateModel()

    # %% Training
    history = model.fit(X_train, y_train, epochs=epochs,validation_data=(X_valid, y_valid), batch_size=batch_size, verbose=2, shuffle=False)
    # %% Plotting
    plot_loss (history, 'LSTM')
    model.save(model_name)
else:
    model = load_model(model_name)
model.summary()



# %% Evaluation
model.reset_states()
# Predictions
X_train_Predict = model.predict(X_train, batch_size=batch_size)
X_train_Predict = scaler.inverse_transform(X_train_Predict)
X_valid_Predict = model.predict(X_valid, batch_size=batch_size)
X_valid_Predict = scaler.inverse_transform(X_valid_Predict)
X_test_Predict = model.predict(X_test, batch_size=batch_size)
X_test_Predict = scaler.inverse_transform(X_test_Predict)
# Actual
y_train = scaler.inverse_transform([y_train])
y_valid = scaler.inverse_transform([y_valid])
y_test = scaler.inverse_transform([y_test])

# %% calculate root mean squared error
train_mse = tf.keras.losses.mean_squared_error(y_train[0], X_train_Predict[:,0]).numpy()
train_mape = tf.keras.losses.mean_absolute_percentage_error(y_train[0], X_train_Predict[:,0]).numpy()
train_mae = tf.keras.losses.mean_absolute_error(y_train[0], X_train_Predict[:,0]).numpy()
train_msle = tf.keras.losses.mean_squared_logarithmic_error(y_train[0], X_train_Predict[:,0]).numpy()
valid_mse = tf.keras.losses.mean_squared_error(y_valid[0], X_valid_Predict[:,0]).numpy()
valid_mape = tf.keras.losses.mean_absolute_percentage_error(y_valid[0], X_valid_Predict[:,0]).numpy()
valid_mae = tf.keras.losses.mean_absolute_error(y_valid[0], X_valid_Predict[:,0]).numpy()
valid_msle = tf.keras.losses.mean_squared_logarithmic_error(y_valid[0], X_valid_Predict[:,0]).numpy()
test_mse = tf.keras.losses.mean_squared_error(y_test[0], X_test_Predict[:,0]).numpy()
test_mape = tf.keras.losses.mean_absolute_percentage_error(y_test[0], X_test_Predict[:,0]).numpy()
test_mae = tf.keras.losses.mean_absolute_error(y_test[0], X_test_Predict[:,0]).numpy()
test_msle = tf.keras.losses.mean_squared_logarithmic_error(y_test[0], X_test_Predict[:,0]).numpy()
print('Train Score: %.2f MSE' % (train_mse))
print('Train Score: %.2f RMSE' % (math.sqrt(train_mse)))
print('Train Score: %.2f MAPE' % (train_mape))
print('Train Score: %.2f MAE' % (train_mae))
print('Train Score: %.2f MSLE' % (train_msle))
print('Valid Score: %.2f MSE' % (valid_mse))
print('Valid Score: %.2f RMSE' % (math.sqrt(valid_mse)))
print('Valid Score: %.2f MAPE' % (valid_mape))
print('Valid Score: %.2f MAE' % (valid_mae))
print('Valid Score: %.2f MSLE' % (valid_msle))
print('Test Score: %.2f MSE' % (test_mse))
print('Test Score: %.2f RMSE' % (math.sqrt(test_mse)))
print('Test Score: %.2f MAPE' % (test_mape))
print('Test Score: %.2f MAE' % (test_mae))
print('Test Score: %.2f MSLE' % (test_msle))


# %% Plotting Predictions
# shift train predictions for plotting
trainPredictPlot = np.empty_like(scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(X_train_Predict)+look_back, :] = X_train_Predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(X_train_Predict)+(look_back*2)+1:len(scaled)-1, :] = X_test_Predict
# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.title(model_type)
plt.ylabel('Daily, {}'.format(Data_Column))
plt.plot(scaler.inverse_transform(scaled), label='Dataset')
plt.plot(trainPredictPlot, label='Training Prediction')
plt.plot(testPredictPlot, label='Testing Prediction')
plt.legend(loc='upper left')


# %% Plotting Multi-Step Forecasting
# Make prediction for new data
# predictions = model.predict() #this creates states
future_x = []
future_y = []
currentStep = scaled[-look_back:] #last step from the previous prediction
# model.reset_states()
for i in range(future_pred_count):
    predicted = model.predict(currentStep.reshape(batch_size,look_back,1)) #get the next step
    future_x.append(currentStep) #store the future steps \
    future_y.append(predicted)
    future_y[i] = np.reshape(scaler.inverse_transform(future_y[i]), 1)
    currentStep = np.concatenate((currentStep[1:], predicted), axis=0).astype(np.float32)
future_predictions_x = pd.DataFrame(np.array(future_x).reshape(future_pred_count,look_back))
future_predictions_y = pd.DataFrame(future_y,columns=['Future'])
# print(future_predictions_x)
print(future_predictions_y)
# Plot history and future
def plot_multi_step(history, prediction1):
    plt.figure(figsize=(15, 6))
    range_history = len(history)
    range_future = list(range(range_history, range_history + len(prediction1)))
    plt.plot(np.arange(range_history), np.array(history), label='History')
    plt.plot(range_future, np.array(prediction1),label='Forecasted with {}'.format(model_type))
    plt.legend(loc='upper right')
    plt.xlabel('Time step')
    plt.ylabel('Daily, {}'.format(Data_Column))
    plt.show()
plot_multi_step(df[-60:], future_predictions_y)
