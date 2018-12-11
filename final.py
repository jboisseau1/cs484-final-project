import pandas as pd
import numpy as np
import gc

import warnings
warnings.filterwarnings("ignore")

def addFeatures(df):
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['healsAndBoosts'] = df['heals']+df['boosts']
    df['totalDistance'] = df['walkDistance']+df['rideDistance']+df['swimDistance']


    df['boostsPerWalkDistance'] = df['boosts']/(df['walkDistance']+1)
    df['boostsPerWalkDistance'].fillna(0, inplace=True)
    df['healsPerWalkDistance'] = df['heals']/(df['walkDistance']+1)
    df['healsPerWalkDistance'].fillna(0, inplace=True)
    df['healsAndBoostsPerWalkDistance'] = df['healsAndBoosts']/(df['walkDistance']+1)
    df['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)


    df['killsPerWalkDistance'] = df['kills']/(df['walkDistance']+1)
    df['killsPerWalkDistance'].fillna(0, inplace=True)


    return df


train = addFeatures(pd.read_csv('inputs/train_V2.csv'))
test = addFeatures(pd.read_csv('inputs/test_V2.csv'))


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=3)

neigh.fit(train[['weaponsAcquired', 'killPlace', 'totalDistance',	'killsPerWalkDistance','healsAndBoostsPerWalkDistance']][:700000], train['winPlacePerc'][:700000])
predcited = neigh.predict(train[['weaponsAcquired', 'killPlace', 'totalDistance',	'killsPerWalkDistance','healsAndBoostsPerWalkDistance']][800000:890000])


from sklearn.metrics import explained_variance_score
EVS= explained_variance_score(train['winPlacePerc'][800000:890000], predcited)
print(EVS)
