#+-------------------------------------------------------------------
# import keras package
#+-------------------------------------------------------------------
import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
# start Params
k_reg = 0.01
mt_optimizer = 'adam'
mt_loss = 'mean_squared_error'
# End Params
mt_model = m1.dl_model_tune(k_reg, mv_X_train, mt_optimizer, mt_loss)

# Wrap the model using KerasRegressor
mt_model = KerasRegressor(build_fn=m1.dl_model_tune, epochs=50, batch_size=32, verbose=1)

# Define the grid of `k_reg` values to search
param_grid = {'k_reg': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}

# Use GridSearchCV to search over the parameter grid
lt_X_train= mv_X_train[:len(mv_y_train)]  # Truncate 'x' to match 'y'
grid = GridSearchCV(estimator=mt_model,param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
print("Params: ",grid.get_params().keys())
grid_result = grid.fit(lt_X_train, mv_y_train)

# Output the best parameters and results
#print(f"Best k_reg: {grid_result.best_params_}")
#print(f"Best MSE: {grid_result.best_score_}")
