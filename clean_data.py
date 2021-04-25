import pandas as pd
import time

start = time.time()
data = pd.read_csv('all_matches.csv')

# Dropping redundant rows
data = data[data['ball'] < 6.1]
data = data[data['innings'] <= 2]


# Deleting duplicate venues
data = data.replace(to_replace=["Punjab Cricket Association IS Bindra Stadium",
                                "Punjab Cricket Association IS Bindra Stadium, Mohali",
                                "Punjab Cricket Association Stadium, Mohali"],
                    value="Punjab Cricket Association Stadium")

data = data.replace(to_replace=['Feroz Shah Kotla', 'M.Chinnaswamy Stadium',  'MA Chidambaram Stadium, Chepauk',
                                'MA Chidambaram Stadium, Chepauk, Chennai', 'Rajiv Gandhi International Stadium, Uppal',
                                'Subrata Roy Sahara Stadium', 'Wankhede Stadium, Mumbai', 'Sardar Patel Stadium, Motera'],
                    value=['Arun Jaitley Stadium', 'M Chinnaswamy Stadium', 'MA Chidambaram Stadium',
                           'MA Chidambaram Stadium', 'Rajiv Gandhi International Stadium',
                           'Maharashtra Cricket Association Stadium', 'Wankhede Stadium', 'Narendra Modi Stadium'])


# Forming data to create csv file of bowler data
data_bowl = data
data_bowl['total_runs'] = data_bowl['runs_off_bat'] + data_bowl['extras']
data_bowl['ball'] += data_bowl['wides'] * 100 + data_bowl['noballs'] * 100

new = data_bowl.groupby(['venue', 'innings', 'bowler'], as_index=False).agg({
    'total_runs': 'sum',
    'ball': lambda b: (b<100).count()})
new['economy'] = (new['total_runs'] / new['ball'])*6
new.drop(new.columns[[3, 4]], axis=1, inplace=True)
new.to_csv('clean_data_bowl.csv', index=False)


# Forming data to create csv file of batsmen data
data = data[(data['extras'] < 1) | (data['noballs'] > 0) | (data['byes'] > 0) | (data['legbyes'] > 0)]
new = data.groupby(['venue', 'innings', 'striker'], as_index=False).agg({
    'runs_off_bat': 'sum',
    'ball': 'count'})

new['strike_rate'] = (new['runs_off_bat'] / new['ball'])
new.drop(new.columns[[3, 4]], axis=1, inplace=True)
new.to_csv('clean_data_bat.csv', index=False)
