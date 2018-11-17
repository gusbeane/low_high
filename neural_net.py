import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree as kDTree
from sklearn.neural_network import MLPRegressor

train = 0.75
np.random.seed(162)

# read in data
lres_out = 'lowresolution_properties_xmatch.fits'
hres_out = 'highresolution_properties_xmatch.fits'

lres_t = Table.read(lres_out)
hres_t = Table.read(hres_out)

lres = np.transpose([lres_t['mass'], lres_t['kinetic_energy'], 
                     lres_t['Vmax'], lres_t['Vmax_radius']])

hres = np.transpose([hres_t['mstar'], hres_t['SFR']])

# choose a sample to train on and to test on
num_to_choose = int(len(lres)*train)
keys_train = np.random.choice(range(len(lres)), size=num_to_choose, replace=False)
keys_test = np.array([i for i in range(len(lres)) if i not in keys_train])

lres_train = lres[keys_train]
hres_train = hres[keys_train]
lres_test = lres[keys_test]
hres_test = hres[keys_test]

mstar_train = hres_train[:,0]
mstar_test = hres_test[:,0]

# run the neural network aka magic
# clf_mstar = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)
clf_mstar = MLPRegressor(max_iter=100000000, learning_rate_init=0.0000001)

# clf_mstar.fit(lres_train.astype('f'), mstar_train.astype('f'))
