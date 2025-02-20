def get_switch_variables(mp_symbol_primary='EURUSD'):  # Make mp_symbol_primary a parameterke mp_symbol_primary a parameter
    
   """
      Returns a dictionary of configuration parameters for data loading and model training.    Returns a dictionary of configuration parameters for data loading and model training.

      Args:
         mp_symbol_primary (str): The primary trading symbol (e.g., 'EURUSD'). Defaults to 'EURUSD'.trading symbol (e.g., 'EURUSD'). Defaults to 'EURUSD'.

      Returns:
         dict: A dictionary containing the configuration parameters.ers.
   """ 

    MPDATAFILENAME1 = "tickdata1.csv"a1.csv"
    MPDATAFILENAME2 = "ratesdata1.csv".csv"
    MPDATAFILE1 = mp_symbol_primary + "_" + MPDATAFILENAME1E1 = mp_symbol_primary + "_" + MPDATAFILENAME1
    MPDATAFILE2 = mp_symbol_primary + "_" + MPDATAFILENAME2 + "_" + MPDATAFILENAME2

    return {
        "broker": "METAQUOTES",  # "ICM" or "METAQUOTES"     "broker": "METAQUOTES",  # "ICM" or "METAQUOTES"
        "mp_symbol_primary": mp_symbol_primary, # Use the passed parametermp_symbol_primary": mp_symbol_primary, # Use the passed parameter
        "mp_symbol_secondary": mp_symbol_primary, # Consistent with primary        "mp_symbol_secondary": mp_symbol_primary, # Consistent with primary
        "MPDATAFILE1": MPDATAFILE1,
        "MPDATAFILE2": MPDATAFILE2, "MPDATAFILE2": MPDATAFILE2,
        "DFNAME1": "df_rates1",
        "DFNAME2": "df_rates2",        "DFNAME2": "df_rates2",
        "mp_data_cfg_usedata": 'loadfilerates',  # 'loadapiticks' or 'loadapirates' or 'loadfileticks' or 'loadfilerates'mp_data_cfg_usedata": 'loadfilerates',  # 'loadapiticks' or 'loadapirates' or 'loadfileticks' or 'loadfilerates'
        "mp_data_rows": 2000,  # Number of rows to fetch for display
        "mp_data_rowcount": 10000,  # Number of rows to fetch for processing        "mp_data_rowcount": 10000,  # Number of rows to fetch for processing
        # Model Tuningdel Tuning
        "ONNX_save": False,  # Save the model in ONNX format
        "mp_ml_show_plot": False,  # Show plots during training "mp_ml_show_plot": False,  # Show plots during training
        "mp_ml_hard_run": True,  # Perform a full/rigorous training run        "mp_ml_hard_run": True,  # Perform a full/rigorous training run
        "mp_ml_tunemode": True,  # Enable hyperparameter tuningable hyperparameter tuning
        "mp_ml_tunemodeepochs": True,  # Tune the number of epochs # Tune the number of epochs
        "mp_ml_Keras_tuner": 'hyperband',  # Hyperparameter optimization algorithm ('hyperband', 'randomsearch', 'bayesian', 'skopt', 'optuna') optimization algorithm ('hyperband', 'randomsearch', 'bayesian', 'skopt', 'optuna')
        "batch_size": 4,  # Batch size for training
        # Model Scaling        # Model Scaling
        "all_modelscale": 2,  # Scaling factor for all model components_modelscale": 2,  # Scaling factor for all model components
        "cnn_modelscale": 2,  # Scaling factor for CNN modelsodels
        "lstm_modelscale": 2,  # Scaling factor for LSTM models
        "gru_modelscale": 2,  # Scaling factor for GRU models
        "trans_modelscale": 2,  # Scaling factor for Transformer modelscaling factor for Transformer models
        "transh_modelscale": 1,  # Scaling factor for Transformer (head) modelsScaling factor for Transformer (head) models
        "transff_modelscale": 4,  # Scaling factor for Transformer (feed-forward) models,  # Scaling factor for Transformer (feed-forward) models
        "dense_modelscale": 2  # Scaling factor for dense layers# Scaling factor for dense layers
    }
"""