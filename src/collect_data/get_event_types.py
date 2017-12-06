'''
Get all possible event types from volunteer and turker interactions
'''

import pandas as pd
import psycopg2 as pg

vol_con = pg.connect(
  database="sidewalk",
  user="sidewalk",
  password="sidewalk",
  host="localhost",
  port="5432")

turk_con = pg.connect(
  database="sidewalk_turker",
  user="sidewalk",
  password="sidewalk",
  host="localhost",
  port="5432")

vol_actions = pd.read_sql(
  '''
      SELECT DISTINCT action
      FROM audit_task_interaction
  ''', vol_con)
vol_actions = vol_actions['action'].tolist()

turk_actions = pd.read_sql(
  '''
      SELECT DISTINCT action
      FROM audit_task_interaction
  ''', turk_con)
turk_actions = turk_actions['action'].tolist()

# combine volunteer and turker action types into unique list
all_actions = set(vol_actions).union(set(turk_actions))
all_actions = list(all_actions)
all_actions.sort()

df = pd.DataFrame(data={'event_type': all_actions})
df.to_csv('../../data/interim/event-types.csv', index=False)

# for each event type, create a feature for total, mean, and std
features_per_action = ['total', 'per_pan_mean', 'per_pan_std']
all_features = [(a + '_' + fa) for a in all_actions for fa in features_per_action]
all_features.sort()

df = pd.DataFrame(data={'feature': all_features})
df.to_csv('../../data/interim/feature-names.csv', index=False)
