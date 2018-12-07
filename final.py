import pandas as pd
import numpy as np
import gc

import warnings
warnings.filterwarnings("ignore")

def addFeatures(df):
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']][5:8]
    df['healsAndBoosts'] = df['heals']+df['boosts']
    df['totalDistance'] = df['walkDistance']+df['rideDistance']+df['swimDistance']


    df['boostsPerWalkDistance'] = df['boosts']/(df['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
    df['boostsPerWalkDistance'].fillna(0, inplace=True)
    df['healsPerWalkDistance'] = df['heals']/(df['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
    df['healsPerWalkDistance'].fillna(0, inplace=True)
    df['healsAndBoostsPerWalkDistance'] = df['healsAndBoosts']/(df['walkDistance']+1) #The +1 is to avoid infinity.
    df['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)


    df['killsPerWalkDistance'] = df['kills']/(df['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
    df['killsPerWalkDistance'].fillna(0, inplace=True)
    df['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in df['numGroups']]

    return df


train = addFeatures(pd.read_csv('inputs/train_V2.csv'))
test = addFeatures(pd.read_csv('inputs/test_V2.csv'))


# solos = train[train['numGroups']>50]
# duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
# squads = train[train['numGroups']<=25]


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(train[['weaponsAcquired', 'damageDealt', 'killPlace', 'totalDistance', 'boostsPerWalkDistance',	'healsPerWalkDistance',	'healsAndBoostsPerWalkDistance',	'killsPerWalkDistance']][:20000], train['winPlacePerc'][:20000])

# dist, indices = neigh.kneighbors(train[['weaponsAcquired', 'damageDealt', 'killPlace', 'totalDistance', 'boostsPerWalkDistance',	'healsPerWalkDistance',	'healsAndBoostsPerWalkDistance',	'killsPerWalkDistance']][300:301])
# print(dist,indices)
predcited = neigh.predict(train[['weaponsAcquired', 'damageDealt', 'killPlace', 'totalDistance', 'boostsPerWalkDistance',	'healsPerWalkDistance',	'healsAndBoostsPerWalkDistance',	'killsPerWalkDistance']][30000:30001])
print('Acc:',(predcited/train['winPlacePerc'][30000:30001].values[0]))
