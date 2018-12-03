import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train_V2.csv')
test = pd.read_csv('test_V2.csv')

# solos = train[train['numGroups']>50]
# duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
# squads = train[train['numGroups']<=25]

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']][5:8]
train['healsAndBoosts'] = train['heals']+train['boosts']
train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']


train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
train['boostsPerWalkDistance'].fillna(0, inplace=True)
train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
train['healsPerWalkDistance'].fillna(0, inplace=True)
train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts']/(train['walkDistance']+1) #The +1 is to avoid infinity.
train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)


train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
train['killsPerWalkDistance'].fillna(0, inplace=True)
train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]
#print(train[['team','kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance','DBNOs','headshotKills','weaponsAcquired', 'winPlacePerc']].sort_values(by='winPlacePerc').tail(100))
#he borrado team de X, acordarme de meterlo
X = train[['weaponsAcquired', 'damageDealtNorm', 'killPlace', 'totalDistance','boostsPerWalkDistance','healsPerWalkDistance','healsAndBoostsPerWalkDistance','killsPerWalkDistance', 'winPlacePerc']].tail(300)
X1 = X.iloc[:, 0:8].values
X2 = X.iloc[:, 8].values
X_std = StandardScaler().fit_transform(X1)
print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))
covariance_mat = np.cov(X_std.T)

eig_values, eig_vectors = np.linalg.eig(covariance_mat)

print('Eigenvectors \n%s' %eig_vectors)
print('\nEigenvalues \n%s' %eig_values)

#  Making a list of pairs (autovector, autovalue)
eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]

# Sort the pairs
#eig_pairs.sort(key=lambda x: x[0], reverse=True)

print('Autovalues in descending order:')
for i in eig_pairs:
    print(i[0])

# Calculate variance
tot = sum(eig_values)
var_exp = [(i / tot)*100 for i in sorted(eig_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Representing the variance for each autovalue and the cumulative variance
with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(8), var_exp, alpha=0.5, align='center',
            label='Individual explained variance', color='g')
    plt.step(range(8), cum_var_exp, where='mid', linestyle='--', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal Components')
    plt.legend(loc='best')
    plt.tight_layout()

plt.show()