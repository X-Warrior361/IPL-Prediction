import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Loading data
venues = pd.read_csv('clean_data_bowl.csv')
full_data = pd.read_csv('playersList.csv')
data_bat = pd.read_csv('clean_data_bat.csv')
data_bowl = pd.read_csv('clean_data_bowl.csv')


# Encoding categorical data
venue, player = LabelEncoder(), LabelEncoder()
venues['venue'] = venue.fit_transform(venues['venue'])
full_data['Players'] = player.fit_transform(full_data['Players'])

data_bat['venue'] = venue.transform(data_bat['venue'])
data_bat['striker'] = player.transform(data_bat['striker'])

data_bowl['venue'] = venue.transform(data_bowl['venue'])
data_bowl['bowler'] = player.transform(data_bowl['bowler'])


# Preparing training and test data
final = data_bat.to_numpy()
x, y = final[:, :3], final[:, 3]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Training the batting model
model_bat = RandomForestRegressor(
    n_estimators=750,
    max_depth=6,
)

model_bat.fit(x_train, y_train)
print(mean_absolute_error(y_test, model_bat.predict(x_test)))


# Preparing training and test data
final = data_bowl.to_numpy()
x, y = final[:, :3], final[:, 3]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Training the bowling model
model_bowl = RandomForestRegressor(
    n_estimators=750,
    max_depth=6,
)

model_bowl.fit(x_train, y_train)
print(mean_absolute_error(y_test, model_bowl.predict(x_test)))


# Saving the joblib files
joblib.dump(model_bat, 'model_bat.joblib')
joblib.dump(model_bowl, 'model_bowl.joblib')
joblib.dump(venue, 'venue.joblib')
joblib.dump(player, 'player.joblib')
