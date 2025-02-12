# Initialize the tuner class
   def initialize_tuner(hypermodel_params, train_dataset, val_dataset, test_dataset):
      try:
         print("Creating an instance of the tuner class")
         mt = CMdtuner(
              self.get_hypermodel_params(**hypermodel_params),
         )
         print("Tuner initialized successfully.")
         return mt
      except Exception as e:
         print(f"Error initializing the tuner: {e}")
         raise