import numpy as np
from astropy.table import Table, Column
from scipy.spatial import cKDTree as kDTree
from sklearn.neural_network import MLPClassifier

lres_name = 'lowresolution_properties.csv'
lres_out = 'lowresolution_properties_xmatch.fits'

hres_name = 'highresolution_properties.csv'
hres_out = 'highresolution_properties_xmatch.fits'

lres = Table.read(lres_name, format='ascii.csv', comment='#')
hres = Table.read(hres_name, format='ascii.csv', comment='#')
hres = hres[hres['mstar'] > 0]

hres_pos = np.transpose([hres['pos_x'], hres['pos_y'], hres['pos_z']])
lres_pos = np.transpose([lres['pos_x'], lres['pos_y'], lres['pos_z']])

keys = np.where(hres['mstar'] > 0)[0]
hres_pos = hres_pos[keys]

kdtree = kDTree(lres_pos)
dist, keys = kdtree.query(hres_pos)
lres = lres[keys]

id_column = Column(keys, name='id')
lres.add_column(id_column)
hres.add_column(id_column)

lres.remove_columns(['pos_x', 'pos_y', 'pos_z'])
hres.remove_columns(['pos_x', 'pos_y', 'pos_z'])

hres.write(hres_out)
lres.write(lres_out)

lres.remove_column('id')
hres.remove_column('id')

lres = np.array(lres)
hres = np.array(hres)

lres = np.array(lres.tolist())
hres = np.array(hres.tolist())

np.save('lres_ml.npy', lres)
np.save('hres_ml.npy', hres)
