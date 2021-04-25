import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.preprocessing import LabelEncoder

start = time()
data = pd.read_csv('clean_data_bowl.csv')

venue, striker = LabelEncoder(), LabelEncoder()
data['venue'] = venue.fit_transform(data['venue'])
data['bowler'] = striker.fit_transform(data['bowler'])

final = data.to_numpy()
x, y = final[:, :3], final[:, 3]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


model = RandomForestRegressor()
param_grid = {
            'n_estimators': [600, 700, 750],
            'max_depth': [6, 7, 8, 9, 10, 11, 15], }
'''min_samples_split': [3, 4, 5],
'min_samples_leaf': [5, 6, 7],'''


gs_cv = GridSearchCV(model, param_grid, n_jobs=4)
# Run grid search on training data
gs_cv.fit(x_train, y_train)
# Print optimal hyperparameters
print(gs_cv.best_params_)
# Check model accuracy (up to two decimal places)
mse = mean_absolute_error(y_train, gs_cv.predict(x_train))
print("Training Set Mean Absolute Error: %.2f" % mse)
mse = mean_absolute_error(y_test, gs_cv.predict(x_test))
print("Test Set Mean Absolute Error: %.2f" % mse)

end = time()
print(end - start)
