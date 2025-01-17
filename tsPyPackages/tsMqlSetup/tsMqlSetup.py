#+------------------------------------------------------------------+
#|                                                    tsMqlSetup.pyw
#|                                                    tony shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

# Importing the necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input
import os
import warnings



class tsMqlSetup:  
         def __init__(self,tflog = '2' , warn=ignore', **kwargs):
            # Initialize instance variables or configuration
            self.kwargs = kwargs
            self.tflog = tflog
            self.warn = warn

            
            #Execute commands to set up the environment
            
            # Suppress warnings
            warnings.filterwarnings(self.warn)
            scaler = StandardScaler()
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tflog  # Suppress TensorFlow warnings
            tf.compat.v1.reset_default_graph()  # Ensure compatibility with TensorFlow v1 functions
            print("Tensorflow Version", tf.__version__)

      
            # Check GPU availability and configure memory growth if a GPU is available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)