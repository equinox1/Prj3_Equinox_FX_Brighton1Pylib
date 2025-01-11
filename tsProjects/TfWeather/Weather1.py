import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#This tutorial will just deal with hourly predictions, so start by sub-sampling the data from 10-minute intervals to one-hour intervals:

# Define a custom path
mp_modelpath = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsProjects/TfWeather/datasets"
custom_cache_dir = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsProjects/TfWeather"  # Replace with your desired directory
custom_subdir = r"/datasets"  # Optional subdirectory


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    cache_dir=custom_cache_dir,
    cache_subdir=custom_subdir,
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
print("csv_path", csv_path)

print(" mp_modelpath", mp_modelpath)
csv_file = mp_modelpath + '/' + 'jena_climate_2009_2016.csv' + '.zip'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print(df.head(10))
else:
    print(f"File {csv_file} does not exist.")


# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]
print("converting date time")
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
#print("df.count()", df.count())
print("df.head(5)", df.head(5))
#Here is the evolution of a few features over time:
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
#plt.show()
# Inspect and cleanup
df.describe().transpose()

# Wind velocity
#One thing that should stand out is the min value of the wind velocity (wv (m/s)) 
# and the maximum value (max. wv (m/s)) columns. This -9999 is likely erroneous.
#There's a separate wind direction column, so the velocity should be greater than zero (>=0). Replace it with zeros:

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()

plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
#plt.show()

#Feature engineering
#Before diving in to build a model, it's important to understand your data and be sure 
# that you're passing the model appropriately formatted data.

#Wind
#The last column of the data, wd (deg)—gives the wind direction in units of degrees. 
# Angles do not make good model inputs: 360° and 0° should be close to each other 
# and wrap around smoothly. Direction shouldn't matter if the wind is not blowing.
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
#plt.show()

#But this will be easier for the model to interpret if you convert the wind direction and velocity columns to a wind vector:

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

#The distribution of wind vectors is much simpler for the model to correctly interpret:

plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')
#plt.show()

#Time
#Similarly, the Date Time column is very useful, but not in this string form. Start by converting it to seconds:
#Taking each 6th record as we need only hourly data, so ignoring every other record(which are on 10 minute level)

timestamp_s = date_time.map(pd.Timestamp.timestamp)

#Similar to the wind direction, the time in seconds is not a useful model input. 
# Being weather data, it has clear daily and yearly periodicity. There are many ways you could deal with periodicity.
#You can get usable signals by using sine and cosine transforms to clear "Time of day" and "Time of year" signals:

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
#plt.show()

#This gives the model access to the most important frequency features. In this case you knew ahead of time
# which frequencies were important.
#If you don't have that information, you can determine which frequencies are important by extracting features with 
#Fast Fourier Transform. To check the assumptions, here is the tf.signal.rfft of the temperature over time.
#Note the obvious peaks at frequencies near 1/year and 1/day:

fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
#plt.show()

#Split the data
#You'll use a (70%, 20%, 10%) split for the training, validation, and test sets. 
# Note the data is not being randomly shuffled before splitting. This is for two reasons:
#It ensures that chopping the data into windows of consecutive samples is still possible.
#It ensures that the validation/test results are more realistic, being evaluated on the data collected after the model was trained.

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

print("train_df.HEAD(5)", train_df.head(5))
print("val_df.HEAD(5)", val_df.head(5))
print("test_df.HEAD(5)", test_df.head(5))

#Normalize the data
#It is important to scale features before training a neural network. Normalization is a common way of doing this 
#scaling: subtract the mean and divide by the standard deviation of each feature.
#The mean and standard deviation should only be computed using the training data so that the models have no 
# access to the values in the validation and test sets.
#It's also arguable that the model shouldn't have access to future values in the training set when training,
# and that this normalization should be done using moving averages. That's not the focus of this tutorial, 
# and the validation and test sets ensure that you get (somewhat) honest metrics. So, in the interest of simplicity this 
# tutorial uses a simple average.

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#Now, peek at the distribution of the features. Some features do have long tails, 
#but there are no obvious errors like the -9999 wind velocity value.
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
#plt.show()

#Data windowing
#The models in this tutorial will make a set of predictions based on a window of consecutive samples from the data.
#The main features of the input windows are:

#The width (number of time steps) of the input and label windows.
#The time offset between them.
#Which features are used as inputs, labels, or both.
#This tutorial builds a variety of models (including Linear, DNN, CNN and RNN models), and uses them for both:

#Single-output, and multi-output predictions.
#Single-time-step and multi-time-step predictions.
#This section focuses on implementing the data windowing so that it can be reused for all of those models.

#Depending on the task and type of model you may want to generate a variety of data windows. Here are some examples:

#For example, to make a single prediction 24 hours into the future, given 24 hours of history, you might define a window like this:
#One prediction 24 hours into the future.
#========================================
#A model that makes a prediction one hour into the future, given six hours of history, would need a window like this:
#One prediction one hour into the future.
#========================================
#The rest of this section defines a WindowGenerator class. This class can:

#Handle the indexes and offsets as shown in the diagrams above.
#Split windows of features into (features, labels) pairs.
#Plot the content of the resulting windows.

#Tf.data.Dataset
#Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.
#1. Indexes and offsets
#Start by creating the WindowGenerator class. The __init__ method includes all the necessary logic for the input and label indices.
#It also takes the training, evaluation, and test DataFrames as input. These will be converted to tf.data.Datasets of windows later.

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


#Here is code to create the 2 windows shown in the diagrams at the start of this section:


w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['T (degC)'])
w1

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['T (degC)'])
w2

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels


WindowGenerator.split_window = split_window



# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

#Typically, data in TensorFlow is packed into arrays where the outermost index is across examples 
#(the "batch" dimension). The middle indices are the "time" or "space" (width, height) dimension(s). 
#The innermost indices are the features.The code above took a batch of three 7-time step windows with 
#19 features at each time step. It splits them into a batch of 6-time step 19-feature inputs, 
#and a 1-time step 1-feature label. 
# The label only has one feature because the WindowGenerator 
#was initialized with label_columns=['T (degC)']. Initially, this tutorial will build models that predict single output labels.
w2.example = example_inputs, example_labels


def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

w2.plot()


#You can plot the other columns, but the example window w2 configuration only has labels for the T (degC) column.
w2.plot(plot_col='p (mbar)')

plt.show()

#Create tf.data.Datasets
#========================
#Finally, this make_dataset method will take a time series DataFrame and convert it to a tf.data.Dataset of (input_window, label_window) 
#pairs using the tf.keras.utils.timeseries_dataset_from_array function:

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

#The WindowGenerator object holds training, validation, and test data.
#Add properties for accessing them as tf.data.Datasets using the make_dataset method you defined earlier. 
#Also, add a standard example batch for easy access and plotting:

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  #Get and cache an example batch of `inputs, labels` for plotting.
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example




#Now, the WindowGenerator object gives you access to the tf.data.Dataset objects, so you can easily iterate over the data.
#The Dataset.element_spec property tells you the structure, data types, and shapes of the dataset elements.
# Each element is an (inputs, label) pair.
w2.train.element_spec

#Iterating over a Dataset yields concrete batches:

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')


#Single step models
#==================
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])
single_step_window

#The window object creates tf.data.Datasets from the training, validation, and test sets, allowing you to easily iterate over batches of data.
for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

#Baseline
#Before building a trainable model it would be good to have a performance baseline as a point for comparison 
#with the later more complicated models.
#This first task is to predict temperature one hour into the future, given the current value of all features.
#The current values include the current temperature.
#So, start with a model that just returns the current temperature as the prediction, predicting "No change". 
#This is a reasonable baseline since temperature changes slowly. Of course, this baseline will work less well 
#if you make a prediction further in the future.

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

#Instantiate and evaluate this model:
baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

#That printed some performance metrics, but those don't give you a feeling for how well the model is doing.
#The WindowGenerator has a plot method, but the plots won't be very interesting with only a single sample.
#So, create a wider WindowGenerator that generates windows 24 hours of consecutive inputs and labels at a time.
# The new wide_window variable doesn't change the way the model operates. The model still makes predictions 
# one hour into the future based on a single input time step. 
# Here, the time axis acts like the batch axis: 
# each prediction is made independently with no interaction between time steps:

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)'])

wide_window

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

#By plotting the baseline model's predictions, notice that it is simply the labels shifted right by one hour:

wide_window.plot(baseline)
plt.show()

#Linear model
#The simplest model you can build on this sort of data is one that predicts the future temperature

linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

#This tutorial trains many models, so package the training procedure into a function:

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

#Train the model and evaluate its performance:
history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val, return_dict=True)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', linear(wide_window.example[0]).shape)

wide_window.plot(linear)

plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)

plt.show()
