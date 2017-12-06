'''
Get user and condition ids for each train/test session
'''

import pandas as pd

# volunteers
vol_data = pd.read_csv('../../data/raw/vol-accuracy.csv')
vol_users = vol_data[['condition.id', 'worker1']]
vol_users = vol_users.rename(
  columns={'condition.id': 'condition_id', 'worker1': 'worker_id'}
)
vol_users = vol_users.drop_duplicates()
vol_users = vol_users.dropna()

def type_of_vol_id(x):
  '''
  The volunteer worker1 column is a mix of user ids and 
  ip addresses for users who weren't signed in. This 
  determines what type of id is in a given row.
  '''
  if len(x) == 36:
      return 'user_id'
  if '.' in x and '-' not in x:
      return 'ip_address'
  else:
      return 'other'

vol_users['worker_id_type'] = vol_users['worker_id'].apply(type_of_vol_id)
vol_users.to_csv('../../data/interim/vol-ids.csv', index=False)

# turkers
turk_data = pd.read_csv('../../data/raw/turk-accuracy.csv')
turk_users = turk_data[['condition.id', 'worker1']]
turk_users = turk_users.rename(
  columns={'condition.id': 'condition_id', 'worker1': 'worker_id'}
)
turk_users = turk_users.drop_duplicates()
turk_users = turk_users.dropna()
turk_users.to_csv('../../data/interim/turk-ids.csv', index=False)
