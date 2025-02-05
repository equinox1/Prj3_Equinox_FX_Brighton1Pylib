# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
# Import dataclasses for data manipulation
import pandas as pd
from dataclasses import dataclass
# Import TensorFlow for machine learning
import tensorflow as tf
import onnx
import tf2onnx
import onnxruntime as ort
import onnxruntime.backend as backend
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
from onnx import checker
import warnings
from numpy import concatenate
# Import equinox functionality
from tsMqlConnect import CMqlinit, CMqlBrokerConfig
from tsMqlDataProcess import CMqldataprocess
from tsMqlDataLoader import CMMarketDataParams
from tsMqlSetup import CMqlSetup, CMqlEnvData, CMqlEnvML, CMqlEnvGlobal
from tsMqlML import CMqlmlsetup, CMqlWindowGenerator
from tsMqlMLTune import CMdtuner
from tsMqlReference import CMqlTimeConfig
from sklearn.preprocessing import MinMaxScaler ,StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer


obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore')
strategy = obj1_CMqlSetup.get_computation_strategy()
mp_ml_show_plot=False

broker = "METAQUOTES" # "ICM" or "METAQUOTES"
mp_symbol_primary='EURUSD'
MPDATAFILE1 =  "tickdata1.csv"
MPDATAFILE2 =  "ratesdata1.csv"

def main():
   with strategy.scope():
      feature_scaler = MinMaxScaler()
      label_scaler = MinMaxScaler()
      
      # +-------------------------------------------------------------------
      # STEP: Load Reference class and time variables
      # +-------------------------------------------------------------------
      obj1_CMqlTimeConfig = CMqlTimeConfig(basedatatime='SECONDS', loadeddatatime='MINUTES')
      MINUTES, HOURS, DAYS, TIMEZONE, TIMEFRAME, CURRENTYEAR, CURRENTDAYS, CURRENTMONTH = obj1_CMqlTimeConfig.get_current_time(obj1_CMqlTimeConfig)
      print("MINUTES:",MINUTES, "HOURS:",HOURS, "DAYS:",DAYS, "TIMEZONE:",TIMEZONE)
      
      mp_symbol_primary =  str(obj1_CMqlTimeConfig.TIME_CONSTANTS['SYMBOLS'][0])
      print("mp_symbol_primary:",mp_symbol_primary) 
      TIMEFRAME = obj1_CMqlTimeConfig.TIME_CONSTANTS['TIMEFRAME']['H4'] # override as M1 needs checking
      
      # model api settings
      dataenv = CMqlEnvData()
      mlenv = CMqlEnvML()
      globalenv = CMqlEnvGlobal()

      data_params = dataenv.get_data_params()
      ml_params = mlenv.get_ml_params()
      global_params = globalenv.get_global_params()


      print("dataenv:",dataenv)
      print("mlenv:",mlenv)
      print("globalenv:",globalenv)

      mp_ml_data_type ='M1'
      # +-------------------------------------------------------------------
      # STEP: CBroker Login
      # +-------------------------------------------------------------------
      # initialize the broker
      
      obj1_CMqlBrokerConfig=CMqlBrokerConfig(lpbroker=broker, mp_symbol_primary=mp_symbol_primary, MPDATAFILE1=MPDATAFILE1, MPDATAFILE2=MPDATAFILE1)
      broker_config, mp_symbol_primary, mp_symbol_secondary, mp_shiftvalue, mp_unit = obj1_CMqlBrokerConfig.initialize_mt5(broker, obj1_CMqlTimeConfig)
      # Login to the broker
      obj2_CMqlBrokerConfig=obj1_CMqlBrokerConfig.login_mt5(broker_config)
      print("Broker Login:",obj2_CMqlBrokerConfig)
      file_path = broker_config['MKFILES']
      MPDATAPATH = broker_config['MPDATAPATH']
      
      mp_data_rows = 1000
      # +-------------------------------------------------------------------
      # STEP: Data Preparation and Loading
      # +-------------------------------------------------------------------
      # Set up dataset
      
      obj1_CMqldataprocess = CMqldataprocess(dataenv, mlenv, globalenv)
      mp_data_history_size = dataenv.mp_data_history_size
      print("CURRENTYEAR:",CURRENTYEAR, "CURRENTYEAR-mp_data_history_size",CURRENTYEAR-mp_data_history_size,"CURRENTDAYS:",CURRENTDAYS, "CURRENTMONTH:",CURRENTMONTH,"TIMEZONE:",TIMEZONE)
      #data from date to current date
     
      # Load tick data from MQL and FILE
      obj1_params = CMMarketDataParams(
      api_ticks=dataenv.mp_data_loadapiticks,
      api_rates=dataenv.mp_data_loadapirates,
      file_ticks=dataenv.mp_data_loadfileticks, 
      file_rates=dataenv.mp_data_loadfilerates,
      dfname1=dataenv.mv_data_dfname1, 
      dfname2=dataenv.mv_data_dfname2, 
      utc_from=None,
      symbol_primary=mp_symbol_primary, 
      rows=dataenv.mp_data_rows, 
      rowcount=dataenv.mp_data_rowcount,
      command_ticks=dataenv.mp_data_command_ticks, 
      command_rates=dataenv.mp_data_command_rates,
      data_path=MPDATAPATH, 
      file_value1=broker_config['MPFILEVALUE1'], 
      file_value2=broker_config['MPFILEVALUE2'],
      timeframe=TIMEFRAME
      )

      mv_data_utc_from = obj1_params.set_mql_timezone(CURRENTYEAR-mp_data_history_size, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
      mv_data_utc_to = obj1_params.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
      print("UTC From:",mv_data_utc_from)
      print("UTC To:",mv_data_utc_to)
      

      """

      mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = obj1_params.load_data_from_mql(obj1_CMqldataprocess, obj1_params)
      
      #wrangle the data merging and transforming time to numeric
      if len(mv_tdata1apiticks) > 0:  
            mv_tdata1apiticks = obj1_CMqldataprocess.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="tickobj1_CMqlSetup", filter_int=False, filter_flt=False, filter_obj=False, filter_dobj1_CMqlTimeConfigi=False, filter_dobj1_CMqlTimeConfigf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
      if len(mv_tdata1apirates) > 0:
            mv_tdata1apirates = obj1_CMqldataprocess.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rateobj1_CMqlSetup", filter_int=False, filter_flt=False, filter_obj=False, filter_dobj1_CMqlTimeConfigi=False, filter_dobj1_CMqlTimeConfigf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
      if len(mv_tdata1loadticks) > 0:
            mv_tdata1loadticks = obj1_CMqldataprocess.wrangle_time(mv_tdata1loadticks, mp_unit, mp_filesrc="ticks2", filter_int=False, filter_flt=False, filter_obj=False, filter_dobj1_CMqlTimeConfigi=False, filter_dobj1_CMqlTimeConfigf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
      if len(mv_tdata1loadrates) > 0:
            mv_tdata1loadrates = obj1_CMqldataprocess.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dobj1_CMqlTimeConfigi=False, filter_dobj1_CMqlTimeConfigf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
                
      # Create labels
      mv_tdata1apiticks = obj1_CMqldataprocess.create_label_wrapper(
            df=mv_tdata1apiticks,
            bid_column="T1_Bid_Price",
            ask_column="T1_Ask_Price",
            column_in="T1_Bid_Price",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R1_Open",
            high_column="R1_High",
            low_column="R1_Low",
            close_column="R1_Close",
            run_mode=1,
            **common_ml_params
        )

      mv_tdata1apirates = obj1_CMqldataprocess.create_label_wrapper(
            df=mv_tdata1apirates,
            bid_column="R1_Bid_Price",
            ask_column="R1_Ask_Price",
            column_in="R1_Close",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R1_Open",
            high_column="R1_High",
            low_column="R1_Low",
            close_column="R1_Close",
            run_mode=2,
            **common_ml_params
        )

      mv_tdata1loadticks = obj1_CMqldataprocess.create_label_wrapper(
            df=mv_tdata1loadticks,
            bid_column="T2_Bid_Price",
            ask_column="T2_Ask_Price",
            column_in="T2_Bid_Price",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R2_Open",
            high_column="R2_High",
            low_column="R2_Low",
            close_column="R2_Close",
            run_mode=3,
            **common_ml_params
        )

      mv_tdata1loadrates = obj1_CMqldataprocess.create_label_wrapper(
            df=mv_tdata1loadrates,
            bid_column="R2_Bid_Price",
            ask_column="R2_Ask_Price",
            column_in="R2_Close",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R2_Open",
            high_column="R2_High",
            low_column="R2_Low",
            close_column="R2_Close",
            run_mode=4,
            **common_ml_params
        )

        # Display the data
      obj1_CMqldataprocess.run_mql_print(mv_tdata1apiticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
      obj1_CMqldataprocess.run_mql_print(mv_tdata1apirates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
      obj1_CMqldataprocess.run_mql_print(mv_tdata1loadticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
      obj1_CMqldataprocess.run_mql_print(mv_tdata1loadrates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")

      # copy the data for config selection
      data_sources = [mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates]
      data_copies = [data.copy() for data in data_sources]
      mv_tdata2a, mv_tdata2b, mv_tdata2c, mv_tdata2d = data_copies
      # Define a mapping of configuration values to data variables
      data_mapping = {
            'loadapiticks': mv_tdata2a,
            'loadapirates': mv_tdata2b,
            'loadfileticks': mv_tdata2c,
            'loadfilerates': mv_tdata2d
      }

      # Check the switch of which file to use
      if mp_data_cfg_usedata in data_mapping:
            print(f"Using {mp_data_cfg_usedata.replace('load', '').replace('api', 'API ').replace('file', 'File ').replace('ticks', 'Tick data').replace('rates', 'Rates data')}")
            mv_tdata2 = data_mapping[mp_data_cfg_usedata]
      else:
            print("Invalid data configuration")
            mv_tdata2 = None

      # print shapes of the data
      print("SHAPE: mv_tdata2 shape:", mv_tdata2.shape)

      # +-------------------------------------------------------------------
      # STEP: Normalize the data
      # +-------------------------------------------------------------------
      # Normalize the 'Close' column
      scaler = MinMaxScaler()
      mp_ml_custom_input_keyfeat_list = list(mp_ml_custom_input_keyfeat) 
      mp_ml_custom_input_keyfeat_scaled = [feat + '_Scaled' for feat in mp_ml_custom_input_keyfeat_list]
      
      mv_tdata2[mp_ml_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2[mp_ml_custom_input_keyfeat_list])
      print("print Normalise")
      obj1_CMqldataprocess.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
      print("End of Normalise print")
      print("mv_tdata2.shape",mv_tdata2.shape)

      # +------------------------------------------------------------------
      # STEP: remove datetime dtype to numeric from the data
      # +-------------------------------------------------------------------
      #if len(mv_tdata2) > 0:
      #    mv_tdata2 = obj1_CMqldataprocess.wrangle_time(mv_tdata2, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dobj1_CMqlTimeConfigi=False, filter_dobj1_CMqlTimeConfigf=True, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=False)

      #obj1_CMqldataprocess.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
      # +-------------------------------------------------------------------
      # STEP: add The time index to the data
      # +-------------------------------------------------------------------
      # Check the first column
      first_column = mv_tdata2.columns[0]
      print("PRE INDEX: Count: ",len(mv_tdata2))

      # Ensure no missing values, duplicates, or type issues
      mv_tdata2[first_column] = mv_tdata2[first_column].fillna('Unknown')  # Handle NaNs
      mv_tdata2[first_column] = mv_tdata2[first_column].astype(str)  # Uniform type
      mv_tdata2[first_column] = mv_tdata2[first_column].str.strip()  # Remove whitespaces

      # Set the first column as index
      mv_tdata2.set_index(first_column, inplace=True)
      mv_tdata2=mv_tdata2.dropna()
      print("POST INDEX: Count: ",len(mv_tdata2))

      # +-------------------------------------------------------------------
      # STEP: set the dataset to just the features and the label and sort by time
      # +-------------------------------------------------------------------
      if mp_data_data_label == 1:
            mv_tdata2 = mv_tdata2[[list(mp_ml_custom_input_keyfeat_scaled)[0]]]
            obj1_CMqldataprocess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")
            
      elif mp_data_data_label == 2:
            mv_tdata2 = mv_tdata2[[mv_tdata2.columns[0]] + [list(mp_ml_custom_input_keyfeat_scaled)[0]]]
            # Ensure the data is sorted by time
            mv_tdata2 = mv_tdata2.sort_index()
            obj1_CMqldataprocess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

      elif mp_data_data_label == 3:
            # Ensure the data is sorted by time use full dataset
            mv_tdata2 = mv_tdata2.sort_index()
            obj1_CMqldataprocess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

      # +-------------------------------------------------------------------
      # STEP: Generate X and y from the Time Series
      # +-------------------------------------------------------------------
      # At thsi point the normalised data columns are split across the X and Y data
      m1 = CMqlmlsetup() # Create an instance of the class
      # 1: 24 HOURS/24 HOURS prediction window
      print("1: MINUTES data entries per time frame: MINUTES:", MINUTES, "HOURS:", MINUTES * 60, "DAYS:", MINUTES * 60 * 24)
      timeval = MINUTES * 60 # hours
      pasttimeperiods = 24
      futuretimeperiods = 24
      predtimeperiods = 1
      features_count = len(mp_ml_custom_input_keyfeat)  # Number of features in input
      labels_count = len(mp_ml_custom_output_label)  # Number of labels in output
      batch_size = mp_ml_batch_size

      print("timeval:",timeval, "pasttimeperiods:",pasttimeperiods, "futuretimeperiods:",futuretimeperiods, "predtimeperiods:",predtimeperiods)
      past_width = pasttimeperiods * timeval
      future_width = futuretimeperiods * timeval
      pred_width = predtimeperiods * timeval
      print("past_width:", past_width, "future_width:", future_width, "pred_width:", pred_width)

      # Create the input features (X) and label values (y)
      print("list(mp_ml_custom_input_keyfeat_scaled)", list(mp_ml_custom_input_keyfeat_scaled))

      # STEP: Create input (X) and label (Y) tensors Ensure consistent data shape
      # Create the input (X) and label (Y) tensors Close_scaled is the feature to predict and Close last entry in future the label
      mv_tdata2_X, mv_tdata2_y = m1.create_Xy_time_windows3(mv_tdata2, past_width, future_width, target_column=list(mp_ml_custom_input_keyfeat_scaled), feature_column=list(mp_ml_custom_input_keyfeat))
      print("mv_tdata2_X.shape", mv_tdata2_X.shape, "mv_tdata2_y.shape", mv_tdata2_y.shape)

      # Scale the Y labels
      mv_tdata2_y = scaler.transform(mv_tdata2_y.reshape(-1, 1))  # Transform Y values
      # +-------------------------------------------------------------------
      # STEP: Split the data into training and test sets Fixed Partitioning
      # +-------------------------------------------------------------------
      # Batch size alignment fit the number of rows as whole number divisible by the batch size to avoid float errors
      batch_size = mp_ml_batch_size
      precountX = len(mv_tdata2_X)
      precounty = len(mv_tdata2_y)
      mv_tdata2_X,mv_tdata2_y = m1.align_to_batch_size(mv_tdata2_X,mv_tdata2_y, batch_size)
      print(f"Aligned data: X shape: {mv_tdata2_X.shape}, Y shape: {mv_tdata2_y.shape}")

      # Check the number of rows
      print("Batch size alignment: mv_tdata2_X shape:", mv_tdata2_X.shape,"Precount:",precountX,"Postcount:",len(mv_tdata2_X))
      print("Batch size alignment: mv_tdata2_y shape:", mv_tdata2_y.shape,"Precount:",precounty,"Postcount:",len(mv_tdata2_y))

      # Split the data into training, validation, and test sets

      # STEP: Split data into training, validation, and test sets
      X_train, X_temp, y_train, y_temp = train_test_split(mv_tdata2_X,mv_tdata2_y, test_size=(mp_ml_validation_split + mp_ml_test_split), shuffle=False)
      X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(mp_ml_test_split / (mp_ml_validation_split + mp_ml_test_split)), shuffle=False)

      print(f"Training set: X_train: {X_train.shape}, y_train: {y_train.shape}")
      print(f"Validation set: X_val: {X_val.shape}, y_val: {y_val.shape}")
      print(f"Test set: X_test: {X_test.shape}, y_test: {y_test.shape}")
      
      # +-------------------------------------------------------------------
      # STEP: convert numpy arrays to TF datasets
      # +-------------------------------------------------------------------
      # initiate the object using a window generatorwindow is not  used in this model Parameters
      tf_batch_size = mp_ml_batch_size

      train_dataset = m1.create_tf_dataset(X_train, y_train, batch_size=tf_batch_size, shuffle=True)
      val_dataset = m1.create_tf_dataset(X_val, y_val, batch_size=tf_batch_size, shuffle=False)
      test_dataset = m1.create_tf_dataset(X_test, y_test, batch_size=tf_batch_size, shuffle=False)
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
      input_keras_batch=mp_ml_batch_size
      input_def_keras_batch=None
      # Get the input shape for the model
      input_rows_X=len(X_train)
      input_rows_y=len(y_train)
      input_batch_size=mp_ml_batch_size
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
      output_label_shape = (output_label, mp_ml_custom_output_label_count)
      print(f"Input shape for model: {input_shape}, Output shape for model: {output_label_shape}")
      # +-------------------------------------------------------------------
      # STEP: Tune best model Hyperparameter tuning and model setup
      # +-------------------------------------------------------------------
      base_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projecobj1_CMqlTimeConfigodelsequinox/equinrun/PythonLib/tsModelData/"
      project_name = "prjEquinox1_prod.keras"
      subdir = os.path.join(base_path, 'tshybrid_ensemble_tuning_prod', str(1))

      print("base_path:", base_path, "project_name:", project_name, "subdir:", subdir)

      t1 = CMqlmlsetup(
            input_shape=(input_timesteps, input_features),
            today_date=date.today().strftime('%Y-%m-%d %H:%M:%S'),
            random_seed=np.random.randint(0, 1000),
            base_path=base_path,
            project_name=project_name,
            subdir=subdir  # Use the already defined subdir instead of recomputing it
      )
      # +-------------------------------------------------------------------
      # STEP:Run the Tuner to find the best model configuration
      # +-------------------------------------------------------------------
      # Run the tuner to find the best model configuration Load hyperparameters
      h1 = CMdtunerHyperModel()
      hypermodel_params = h1.get_hypermodel_params()

      # Log the configuration
      log_config(hypermodel_params)

      # Initialize tuner
      mt = initialize_tuner(
            hypermodel_params=hypermodel_params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
      )

      ## Run the tuner to find the best model configuration
      print("Running Main call to tuner")
      mt.tuner.search_space_summary()
      # Check and load the model
      best_model = mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')
      print ("Best model loaded successfully evaluated as ",best_model)
      # If no model or hard run then run the search
      if best_model is None or mp_ml_hard_run:
         print("Running the tuner search")
         mt.run_search()
         print("Tuner search completed")
         print("Exporting the best model")
         mt.export_best_model(ftype='tf')
         print("Best model exported")
         # Reload the best model after exporting
         best_model = mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')
      else:
         print("Existing Best model loaded successfully.")

      # +-------------------------------------------------------------------
      # STEP: Train and evaluate the best model
      # +-------------------------------------------------------------------

      # Model Training
      print("Training the best model...")
      best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=mp_ml_tf_param_epochs,
                batch_size=mp_ml_batch_size
            )
      print("Training completed.")

      # Model Evaluation
      print("Evaluating the model...")
      val_metrics = best_model.evaluate(val_dataset, verbose=0)
      test_metrics = best_model.evaluate(test_dataset, verbose=0)
      print(f"Validation Metrics - Loss: {val_metrics[0]}, Accuracy: {val_metrics[1]}")
      print(f"Test Metrics - Loss: {test_metrics[0]}, Accuracy: {test_metrics[1]}")

      # Fit the label scaler on the training labels
      label_scaler.fit(y_train.reshape(-1, 1))

      # Predictions and Scaling
      print("Running predictions and scaling...")
      predicted_fx_price = best_model.predict(test_dataset)
      predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)

      real_fx_price = label_scaler.inverse_transform(y_test)
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
      plt.show()
      print("Plot Model saved to ", mp_ml_base_path + '/' + 'plot.png')

      # +-------------------------------------------------------------------
      # STEP: Save model to ONNX
      # +-------------------------------------------------------------------

      # Save the model to ONNX format
      mp_output_path = mp_ml_data_path + "model_" + mp_symbol_primary + "_" + mp_ml_data_type + ".onnx"
      print(f"output_path: ",mp_output_path)
      onnx_model, _ = tf2onnx.convert.from_keras(best_model[0], opset=self.batch_size)
      onnx.save_model(onnx_model, mp_output_path)
      print(f"model saved to ",mp_output_path)
   
      #Assuming your model has a single input  Convert the model
      print("mp_inputs: ", mp_inputs)
      spec = mp_inputs.shape
      spec = (tf.TensorSpec(spec, tf.float32, name="input"),)
      print("spec: ", spec)
      # Convert the model to ONNX format
      opver = 17
      onnx_model = tf2onnx.convert.from_keras(best_model, input_signature=spec, output_path=mp_output_path, opset=opver)
      print("ONNX Runtime version:", ort.__version__)
      onnx.save_model(onnx_model, mp_output_path)
      print(f"model saved to ", mp_output_path)

      from onnx import checker 
      checker.check_model(best_model[0])
      # finish
      mt5.shutdown()
      print("Finished")
      """
if __name__ == "__main__":
    main()