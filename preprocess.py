import pandas as pd

# multiclass-classification
# win, loss or draw

# Data is centred on the commentary with other data columns
# being reverse-engineered to identify what sort of event it was
# via some regex.
# Therefore, worthwhile to focus on the `text` column which is the commentary only

# Dataset provides a granular view of 9,074 games, totaling 941,009 events from the
# biggest 5 European football (soccer) leagues: England, Spain, Germany, Italy, France from 2011/2012 season
# to 2016/2017 season as of 25.01.2017.
# There are games that have been played during these seasons for which I could not collect detailed data.
# Overall, over 90% of the played games during these seasons have event data.

# idea: use goals, `is_goal` to get the outcome of a match

# import data
df = pd.read_csv(filepath_or_buffer='data/events.csv')

# aggregate data
dict_aggregate = {'text': ' '.join, 'is_goal': 'sum'}
df_commentary = df.groupby(by=['id_odsp', 'event_team', 'opponent'],
                           as_index=False)\
    .agg(dict_aggregate)

# pivot x window
## get two extra columns
#   1. event_team, opposition_team -> from `event_team`
#   2. event_goals, opposition_goals -> from `is_goal`
COLUMNS_KEEP = ['id_odsp', 'event_team', 'is_goal', 'text', 'row_number']

# window function to partition on `id_odsp` and order by `event_team`, return `row_number`
df_commentary['row_number'] = df_commentary.sort_values(['event_team'],
                                                         ascending=True)\
    .groupby(['id_odsp'])\
    .cumcount() + 1

# filter for event and opposition teams, with row_number being 1 and 2 respectively
df_event = df_commentary[COLUMNS_KEEP].query('row_number == 1')
df_opponent = df_commentary[COLUMNS_KEEP].query('row_number == 2')

# rename columns to avoid clashes when joining
df_event = df_event.rename(columns={'is_goal': 'home_goals',
                                    'text': 'home_text'})
df_opponent = df_opponent.rename(columns={'event_team': 'opponent_team',
                                          'is_goal': 'opponent_goals',
                                          'text': 'opponent_text'})

# join them
df_commentary = df_event.merge(right=df_opponent, on='id_odsp')