# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +------------------------------------------------------------------+
# STEP: Platform settings
# +-------------------------------------------------------------------
# gpu and tensor platform
from tsMqlSetup import CMqlSetup
import logging

# Initialize logger
logger = logging.getLogger("Main")
logging.basicConfig(level=logging.INFO)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
# System packages
import os
import pathlib
from pathlib import Path
import json
from datetime import datetime, date
import pytz

# Data packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Machine Learning packages
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# set package options
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
# Equinox environment manager
from tsMqlEnvMgr import CMqlEnvMgr
#Reference class
from tsMqlReference import CMqlRefConfig
# Equinox sub packages
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
# Equinox ML packages
from tsMqlMLTune import CMdtuner
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLProcess import CDMLProcess

# Setup the logging and tensor platform dependencies
obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore', tfdebug=False)
strategy = obj1_CMqlSetup.get_computation_strategy()
# format values
mp_data_tab_rows = 5
mp_data_tab_width = 30

# +-------------------------------------------------------------------
# STEP: End of driving parameters
# +-------------------------------------------------------------------

def main(logger):
    with strategy.scope():

      #-----------------------------------------------------------------
        
        # +----------------------------------------------------------
        # STEP: setup environment
        # +----------------------------------------------------------
        # Usage:env = EnvManager(custom_params={"ml": {"epochs": 20}}) ,print("ML Epochs:", env.get_param("ml", "epochs"))
        # env.override_params({"base": {"learning_rate": 0.005}}) ,print("Updated Learning Rate:", env.get_param("base", "learning_rate"))
        env = CMqlEnvMgr()
        
        params = env.all_params()
        print("All Parameters:", params)
        # get environment parameters
        params_sections = params.keys()
        print("PARAMS SECTIONS:", params_sections)

        base_params = params["base"]
        data_params = params["data"]
        ml_params = params["ml"]
        mltune_params = params["mltune"]
        app_params = params["app"]

        # STEP: Load Reference class and time variables
        # +-------------------------------------------------------------------
        # Time Overrides
        env.override_params({"data": {"mp_data_timeframe": 'H4'}})
        lp_timeframe_name = env.get_param('data', 'mp_data_timeframe')
        logger.info(f"Timeframe: {lp_timeframe_name}")
       
       # Reference class
        reference_config = CMqlRefConfig(loaded_data_type='MINUTE', required_data_type=lp_timeframe_name)
        #Symbol Constants
        PRIMARY_SYMBOL = app_params.get('lp_app_primary_symbol', app_params.get('mp_app_primary_symbol', 'EURUSD'))
        SECONDARY_SYMBOL = app_params.get('lp_app_secondary_symbol', app_params.get('mp_app_primary_symbol','EURCHF'))
        # Time Constants
        UNIT=reference_config.TIME_CONSTANTS["UNIT"]["SECOND"]
        MINUTE = reference_config.get_timevalue('MINUTE')
        HOUR = reference_config.get_timevalue('HOUR')
        DAY = reference_config.get_timevalue('DAY')
        # Current Time Constants
        CURRENTDAY = reference_config.get_current_time()["CURRENTDAY"]
        CURRENTMONTH = reference_config.get_current_time()["CURRENTMONTH"]
        CURRENTYEAR = reference_config.get_current_time()["CURRENTYEAR"]
   
        # Mql Time Constants
        TIMEZONE = reference_config.get_current_time()["TIMEZONE"]
        TIMEFRAME = reference_config.get_current_time()["TIMEFRAME"]
        timeval = HOUR # hours # used in the window creation
        logger.info(f"Timezone: {TIMEZONE}")
        logger.info(f"Timeframe: {TIMEFRAME}") 
       
        # +-------------------------------------------------------------------
        # STEP: CBroker Login
        # +-------------------------------------------------------------------
        print("PARAM HEADER: MP_APP_BROKER:", app_params.get('mp_app_broker'))
        broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
        mqqlobj = broker_config.run_mql_login()
        if mqqlobj is True:
            print("Successfully logged in to MetaTrader 5.")
        else:
            print(f"Failed to login. Error code: {mqqlobj}")
        # +-------------------------------------------------------------------
        # STEP: Data Preparation and Loading
        # +-------------------------------------------------------------------
        # Retrieve broker file paths
        loader_config = CDataLoader()
        process_config = CDataProcess()
        # +-------------------------------------------------------------------
        # STEP: Data loading and processing
        # +-------------------------------------------------------------------
        # Set the data history size
        mp_data_history_size = data_params.get('mp_data_history_size')
        # Set the UTC time for the data
        mv_data_utc_from = loader_config.set_mql_timezone(CURRENTYEAR - mp_data_history_size, CURRENTMONTH, CURRENTDAY, TIMEZONE)
        mv_data_utc_to = loader_config.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAY, TIMEZONE)
        logger.info(f"Main:UTC From: {mv_data_utc_from}")
        logger.info(f"Main:UTC To: {mv_data_utc_to}")
         # Set the data rows
        mp_app_rows = app_params.get('mp_app_rows') ; logger.info(f"Main:Rows: {mp_app_rows}")
        mp_app_rowcount = app_params.get('mp_app_rowcount')
        # Load the data
        loader_config = CDataLoader(lp_utc_from=mv_data_utc_from, lp_utc_to=mv_data_utc_to, lp_timeframe=lp_timeframe_name, lp_app_primary_symbol=PRIMARY_SYMBOL, lp_app_rows=mp_app_rows , lp_app_rowcount=mp_app_rowcount)
        df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = loader_config.load_data()
        # +-------------------------------------------------------------------
        # STEP: Run data process manipulation
        # +-------------------------------------------------------------------
         # Process the data
        df_api_ticks, df_api_rates, df_file_ticks, df_file_rates = process_config.run_wrangle_service(df_api_ticks=df_api_ticks,df_api_rates=df_api_rates,df_file_ticks=df_file_ticks,df_file_rates=df_file_rates,mp_unit=UNIT)
        logger.info(f"Processed Wrangled DataFrames: df_api_ticks.shape {df_api_ticks.shape},df_api_rates.shape {df_api_rates.shape},df_file_ticks.shape {df_file_ticks.shape},df_file_rates.shape {df_file_rates.shape}")
        #set Common Close column
        for df, df_name in [(df_api_ticks, "df_api_ticks"), (df_api_rates, "df_api_rates"), (df_file_ticks, "df_file_ticks"), (df_file_rates, "df_file_rates")]:
            process_config.establish_common_feat_col(df,df_name)
        # Average the columns
        for df, df_name in [(df_api_ticks, "df_api_ticks"), (df_api_rates, "df_api_rates"), (df_file_ticks, "df_file_ticks"), (df_file_rates, "df_file_rates")]:
            process_config.run_average_columns(df,df_name)
        datafile = df
        logger.info(f"Data File Shape: {datafile.shape}")
        
        # +-------------------------------------------------------------------
        # STEP: add The time index to the data
        # +-------------------------------------------------------------------
        datafile = process_config.create_index_column(datafile)
        # Limit the columns to just the time column and all features
        column_features = datafile.columns[1:]
        logger.info(f"Column Features: {column_features}")
        datafile = datafile[[datafile.columns[0]] + list(datafile.columns[1:])]

        # +-------------------------------------------------------------------
        # STEP: Normalise the data
        # +-------------------------------------------------------------------

        # check if the data is stationary    

        # +-------------------------------------------------------------------
        # STEP: Generate X and y from the Time Series
        # +-------------------------------------------------------------------
        # At this point the normalised data columns are split across the X and Y data
               
        # 1: 24 HOURS/24 HOURS prediction window
        logger.info("Creating the 24 hour prediction timeval{timeval},hour {HOUR}")
        mlprocess_config = CDMLProcess()
        past_width, future_width, pred_width, features_count, labels_count, batch_size = mlprocess_config.create_ml_window(timeval=HOUR)
        logger.info(f"Past Width: {past_width}, Future Width: {future_width}, Prediction Width: {pred_width}, Features Count: {features_count}, Labels Count: {labels_count}, Batch Size: {batch_size}")
         
"""

         # Create the input features (X) and label values (y)
         print("list(mp_data_custom_input_keyfeat_scaled)", list(mp_data_custom_input_keyfeat_scaled))

         # STEP: Create input (X) and label (Y) tensors Ensure consistent data shape
         # Create the input (X) and label (Y) tensors Close_scaled is the feature to predict and Close last entry in future the label
        datafile_X,datafile_y = obj1_Mqlmlsetup.create_Xy_time_windows3(mv_tdata2, past_width, future_width, target_column=list(mp_data_custom_input_keyfeat_scaled), feature_column=list(mp_data_custom_input_keyfeat))
         print("mv_tdata2_X.shape",datafile_X.shape, "mv_tdata2_y.shape",datafile_y.shape)
         
         # +-------------------------------------------------------------------
         # STEP: Normalize the Y data
         # +-------------------------------------------------------------------
         # Scale the Y labels
        datafile_y = scaler.transform(mv_tdata2_y.reshape(-1, 1))  # Transform Y values
                  
         # +-------------------------------------------------------------------
         # STEP: Split the data into training and test sets Fixed Partitioning
         # +-------------------------------------------------------------------
         # Batch size alignment fit the number of rows as whole number divisible by the batch size to avoid float errors
         batch_size = gen_environments["tuneenv"].batch_size
         precountX = len(mv_tdata2_X)
         precounty = len(mv_tdata2_y)
        datafile_X,mv_tdata2_y = obj1_Mqlmlsetup.align_to_batch_size(mv_tdata2_X,mv_tdata2_y, batch_size)
         print(f"Aligned data: X shape: {mv_tdata2_X.shape}, Y shape: {mv_tdata2_y.shape}")

         # Check the number of rows
         print("Batch size alignment:datafile_X shape:",datafile_X.shape,"Precount:",precountX,"Postcount:",len(mv_tdata2_X))
         print("Batch size alignment:datafile_y shape:",datafile_y.shape,"Precount:",precounty,"Postcount:",len(mv_tdata2_y))

         # Split the data into training, validation, and test sets

         # STEP: Split data into training, validation, and test sets
         X_train, X_temp, y_train, y_temp = train_test_split(mv_tdata2_X,mv_tdata2_y, test_size=(gen_environments["tuneenv"].validation_split + gen_environments["tuneenv"].test_split), shuffle=False)
         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(gen_environments["tuneenv"].test_split / (gen_environments["tuneenv"].validation_split + gen_environments["tuneenv"].test_split)), shuffle=False)

         print(f"Training set: X_train: {X_train.shape}, y_train: {y_train.shape}")
         print(f"Validation set: X_val: {X_val.shape}, y_val: {y_val.shape}")
         print(f"Test set: X_test: {X_test.shape}, y_test: {y_test.shape}")
         # +-------------------------------------------------------------------
         # STEP: convert numpy arrays to TF datasets
         # +-------------------------------------------------------------------
         # initiate the object using a window generatorwindow is not  used in this model Parameters
         tf_batch_size = gen_environments["tuneenv"].batch_size

         train_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_train, y_train, batch_size=tf_batch_size, shuffle=True)
         val_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_val, y_val, batch_size=tf_batch_size, shuffle=False)
         test_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_test, y_test, batch_size=tf_batch_size, shuffle=False)
         print(f"TF Datasets created: Train: {tf.data.experimental.cardinality(train_dataset).numpy()}, Val: {tf.data.experimental.cardinality(val_dataset).numpy()}, Test: {tf.data.experimental.cardinality(test_dataset).numpy()}")

         
         # +-------------------------------------------------------------------
         # STEP: add tensor values for model input
         # +-------------------------------------------------------------------

         train_shape, val_shape, test_shape = None, None, None

         for dataset, name in zip([train_dataset, val_dataset, test_dataset], ['train', 'val', 'test']):
               for spec in dataset.element_spec:
                  if name == 'train':
                     train_shape = spec.shape
                  elif name == 'val':
                     val_shape = spec.shape
                  elif name == 'test':
                     test_shape = spec.shape

         # +-------------------------------------------------------------------
         # STEP: Final shape summaries
         # +-------------------------------------------------------------------
         # Final summary of shapes
         # STEP: Confirm tensor shapes for the tuner
         input_shape = X_train.shape[1:]  # Shape of a single sample (time_steps, features)
         label_shape = y_train.shape[1:]  # Shape of a single label (future_steps)
         # Example data: shape = (num_samples, time_steps, features) cnn_data = np.random.random((1000, 1440, 1))  # 1000 samples, 1440 timesteps, 1 feature
         # Example data: labels = np.random.random((1000, 1))
         print(f"Full Input shape for model: {X_train.shape}, Label shape for model: {y_train.shape}")
         print(f"No Batch Input shape for model: {input_shape}, Label shape for model: {label_shape}")
         #batch components
         input_keras_batch=gen_environments["tuneenv"].batch_size
         input_def_keras_batch=None
         # Get the input shape for the model
         input_rows_X=len(X_train)
         input_rows_y=len(y_train)
         input_batch_size=gen_environments["tuneenv"].batch_size
         input_batches= X_train.shape[0]
         input_timesteps = X_train.shape[1]
         input_features = X_train.shape[2]
         # Get the output shape for the model
         output_label=y_train.shape[1]
         output_shape = y_train.shape
         output_features = y_train.shape[1]
         print(f"input_def_keras_batch  {input_def_keras_batch}, input_keras_batch: {input_keras_batch}")
         print(f"Input rows X: {input_rows_X},Input rows y: {input_rows_y} , Input batch_size {input_batch_size}, Input batches: {input_batches}, Input timesteps: {input_timesteps}, Input steps or features: {input_features}")
         print(f"Output label: {output_label}, Output shape: {output_shape}, Output features: {output_features}")
         # pass in the data shape for the model

         input_shape = (input_timesteps, input_features)  
         output_label_shape = (output_label, gen_environments["dataenv"].mp_data_custom_output_label_count)
         print(f"Input shape for model: {input_shape}, Output shape for model: {output_label_shape}")
         # +-------------------------------------------------------------------
         # STEP: Tune best model Hyperparameter tuning and model setup
         # +-------------------------------------------------------------------
         # Hyperparameter configuration
         obj1_TunerParams = CMdtunerHyperModel(
            tuner_params=tunerparams["tunerparams"],
            input_shape=input_shape,
            data_input_shape=input_shape,
            output_shape=output_label_shape,
            input_batch_size=batch_size,
            tunemode=True,
            tunemodeepochs=True,
            model_summary=True,
            batch_size=batch_size,
            epochs=15
         )  
         hypermodel_params = obj1_TunerParams.get_hypermodel_params()
         print("Tuner Parameters:", hypermodel_params)  # Print the tuner parameters

         
         #instansiate tuner claa
         obj1_CMdtuner = CMdtuner(
                     hypermodel_params=hypermodel_params,
                     traindataset=train_dataset,
                     valdataset=val_dataset,
                     testdataset=test_dataset,
                     castmode='float32',
                     
                     )
         #initialize tuner
         obj1_CMdtuner.initialize_tuner()
         
         # Check and load the model
         best_model = obj1_CMdtuner.check_and_load_model(mp_ml_mbase_path, ftype='tf')

         # If no model or a hard run is required, run the search
         runtuner = False

         if best_model is None:
               print("Running the tuner search bo model")
               runtuner = mt.run_search()
               mt.export_best_model(ftype='tf')
         elif mp_ml_hard_run:  
               print("Running the tuner search hard run")
               runtuner = mt.run_search()
               mt.export_best_model(ftype='tf')
         else:
               print("Best model loaded successfully")
               runtuner = True

         # +-------------------------------------------------------------------
         # STEP: Train and evaluate the best model
         # +-------------------------------------------------------------------
         print("Model: Loading file from directory", mp_ml_mbase_path,"Model: filename", mp_ml_project_name)   
         if (load_model := mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')) is not None:
               best_model = load_model
               print("Model: Best model: ", best_model.name)
               print("best_model.summary()", best_model.summary())
               
               # Fit the label scaler on the training labels
               
               # Model Training
               tf.keras.backend.clear_session(free_memory=True)

               print("Training the best model...")
               best_model.fit(
                  train_dataset,
                  validation_data=val_dataset,
                  batch_size=batch_size,
                  epochs=mp_ml_tf_param_epochs
               )
               print("Training completed.")
               
               # Model Evaluation
               print("Evaluating the model...")
               val_metrics = best_model.evaluate(val_dataset, verbose=1)
               test_metrics = best_model.evaluate(test_dataset, verbose=1)

               print(f"Validation Metrics - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}")
               print(f"Test Metrics - Loss: {test_metrics[0]:.4f}, Accuracy: {test_metrics[1]:.4f}")

               # Fit the label scaler on the training labels
               label_scaler.fit(y_train.reshape(-1, 1))

               # Predictions and Scaling
               print("Running predictions and scaling...")
               predicted_fx_price = best_model.predict(test_dataset)
               predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)

               real_fx_price = label_scaler.inverse_transform(y_test.reshape(-1, 1))
               print("Predictions and scaling completed.")
               # +-------------------------------------------------------------------
               # STEP: Performance Check
               # +-------------------------------------------------------------------
               # Evaluation and visualization
               # Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values. 
               # The lower the MSE, the better the model.

               # Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values. 
               # Like MSE, lower values indicate better model performance.

               # R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the
               # dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates a 
               # perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the label
               # variable. Negative values indicate poor model performance.
               # Check for NaN values and handle them
               if np.isnan(real_fx_price).any() or np.isnan(predicted_fx_price).any():
                  print("Warning: NaN values found in input data. Handling NaNs by removing corresponding entries.")
                  mask = ~np.isnan(real_fx_price) & ~np.isnan(predicted_fx_price)
                  real_fx_price = real_fx_price[mask]
                  predicted_fx_price = predicted_fx_price[mask]
               
               mse = mean_squared_error(real_fx_price, predicted_fx_price)
               mae = mean_absolute_error(real_fx_price, predicted_fx_price)
               r2 = r2_score(real_fx_price, predicted_fx_price)
               print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
               print(f"Mean Squared Error: The lower the MSE, the better the model: {mse}")
               print(f"Mean Absolute Error: The lower the MAE, the better the model: {mae}")
               print(f"R2 Score: The closer to 1, the better the model: {r2}")

               plt.plot(real_fx_price, color='red', label='Real FX Price')
               plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
               plt.title('FX Price Prediction')
               plt.xlabel('Time')
               plt.ylabel('FX Price')
               plt.legend()
               plt.savefig(mp_ml_base_path + '/' + 'plot.png')
               if mp_ml_show_plot:
                  plt.show()
               print("Plot Model saved to ", mp_ml_base_path + '/' + 'plot.png')

               if ONNX_save:
                  # Save the model to ONNX
                  
                  # Define the output path
                  mp_output_path = mp_ml_data_path + f"model_{mp_symbol_primary}_{mp_ml_data_type}.onnx"
                  print(f"Output Path: {mp_output_path}")

                  # Convert Keras model to ONNX
                  opset_version = 17  # Choose an appropriate ONNX opset version

                  # Assuming your model has a single input
                  spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]
                  onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)

                  # Save the ONNX model
                  onnx.save_model(onnx_model, mp_output_path)
                  print(f"Model saved to {mp_output_path}")

                  # Verify the ONNX model
                  checker.check_model(onnx_model)
                  print("ONNX model is valid.")

                  # Check ONNX Runtime version
                  print("ONNX Runtime version:", ort.__version__)

                  # finish
                  mt5.shutdown()
                  print("Finished")
         else:
               print("No data loaded")
         mt5.shutdown()
         print("Finished")    
"""
if __name__ == "__main__":
    main(logger)