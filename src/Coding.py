#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pickle and load the data
import pickle
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = open('protein_lig_dataset.pkl','rb')
emp = pickle.load(data)


# In[2]:


type(emp)


# In[3]:


#convert the list into array
emp = np.array(emp)


# In[4]:


import numpy as np


# In[5]:


emp


# In[6]:


#check the file
type(emp)


# In[7]:


#find the shape of the array
np.shape(emp)


# In[8]:


#indexing the file based on the atom types: ligand, protein, water
ligand_pos = emp[:,0:56,:]
protein_pos = emp[:,56:1892,:]
water_pos = emp[:,1892:32210,:]


# In[9]:


#check the shape of the atom ligand
np.shape(ligand_pos)


# In[10]:


#import mdtraj
import mdtraj as mdj


# In[11]:


get_ipython().system('pip install biopandas')


# In[12]:


from biopandas.pdb import PandasPdb
ppdb = PandasPdb()
ppdb.read_pdb('bptf_AU1 (1).pdb')


# In[13]:


df = ppdb.df['ATOM']
df.head()


# In[14]:


df.keys()


# In[15]:


df_pro = df[df['residue_name'] == 'PRO']
df_pro.head()


# In[16]:


type(df_pro)


# In[17]:


ppdb.df['ANISOU'].empty


# In[18]:


df['b_factor'].plot(kind='hist')
plt.title('Distribution of B-Factors')
plt.xlabel('B-factor')
plt.ylabel('count')
plt.show()


# In[19]:


df['b_factor'].plot(kind='line')
plt.title('B-Factors Along the Amino Acid Chain')
plt.xlabel('Residue Number')
plt.ylabel('B-factor in $A^2$')
plt.show()


# In[20]:


df['element_symbol'].value_counts().plot(kind='bar')
plt.title('Distribution of Atom Types')
plt.xlabel('elements')
plt.ylabel('count')
plt.show()


# In[21]:


#filter pdb by distance
reference_point = (-10.871, 61.707, -18.620) #atom 126 
distances = ppdb.distance(xyz=reference_point, records=('ATOM',))


# In[22]:


type(distances)


# In[23]:


dist = distances.to_frame(name='distances')


# In[24]:


dist.describe()


# In[25]:


import pandas as pd


# In[26]:


new_df = df.join(dist)
new_df


# In[27]:


new_df.columns


# In[28]:


ligand = new_df['residue_name'] == 'UNK'
new_df_ligand = new_df[ligand]
print(np.shape(new_df_ligand))
new_df_ligand.head()


# In[29]:


new_df_ligand.describe()


# In[30]:


options = ['HOH','UNK','SOD']
new_df_protein = new_df.loc[~new_df['residue_name'].isin(options)]
print(np.shape(new_df_protein))
new_df_protein.head()


# In[31]:


new_df_protein.describe()


# In[32]:


sequence = ppdb.amino3to1()
sequence.head()


# In[33]:


sequence_list = list(sequence.loc[sequence['chain_id'] == 'A', 'residue_name'])
''.join([ x for x in sequence_list if "?" not in x ])


# In[34]:


for chain_id in sequence['chain_id'].unique():
    print('\nChain ID: %s' % chain_id)
    sequence_list = list(sequence.loc[sequence['chain_id'] == chain_id, 'residue_name'])
    print(''.join([ x for x in sequence_list if "?" not in x ]))


# In[35]:


pdb = mdj.load_pdb('bptf_AU1 (1).pdb')


# In[36]:


pdb


# In[37]:


traj = mdj.Trajectory(emp, pdb.top)
traj


# In[38]:


traj.superpose(pdb, atom_indices = list(range(0,1892)))


# In[39]:


traj.save_dcd('test.dcd')


# In[40]:


# traj.xyz?


# In[41]:


# traj.superpose?


# In[42]:


# mdj.Trajectory?


# In[43]:


data_recenter = traj.xyz
data_recenter_lig = traj.xyz[:,0:56,:]
data_recenter_prot = traj.xyz[:,56:1892,:]
data_recenter_lig_prot = traj.xyz[:,0:1892,:]


# In[44]:


np.shape(data_recenter_lig)


# In[45]:


np.shape(data_recenter_prot)


# In[46]:


np.shape(data_recenter_lig_prot)


# In[47]:


data_flatten = np.array([x.flatten() for x in data_recenter])
data_flatten_lig = np.array([x.flatten() for x in data_recenter_lig])
data_flatten_prot = np.array([x.flatten() for x in data_recenter_prot])
data_flatten_lig_prot = np.array([x.flatten() for x in data_recenter_lig_prot])


# In[48]:


np.shape(data_flatten_lig)


# In[49]:


np.shape(data_flatten_prot)


# In[50]:


np.shape(data_flatten_lig_prot)


# In[51]:


from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from sklearn import manifold

Axes3D

n_points = 1000
X = data_flatten_lig
n_neighbors = 100
n_components = 2

fig = plt.figure(figsize=(15,8))

#standard
t0 = time()
Y1 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'standard').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('standard', t1 - t0))

ax = fig.add_subplot(251)
plt.scatter(Y1[:, 0], Y1[:, 1], c=Y1[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y1)
label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(label[mask])[0]
    
# print(accuracy_score(label, labels))

#ltsa
t0 = time()
Y2 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'ltsa').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('ltsa', t1 - t0))

ax = fig.add_subplot(252)
plt.scatter(Y2[:, 0], Y2[:, 1], c=Y2[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LTSA', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y2)
label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(label[mask])[0]
    
# print(accuracy_score(label, labels))

#hessian
t0 = time()
Y3 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'hessian').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('hessian', t1 - t0))

ax = fig.add_subplot(253)
plt.scatter(Y3[:, 0], Y3[:, 1], c=Y3[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('Hessian LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y3)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)

#modified LLE
t0 = time()
Y4 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'modified').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('modified', t1 - t0))

ax = fig.add_subplot(254)
plt.scatter(Y4[:, 0], Y4[:, 1], c=Y4[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('Modified LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y4)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)

#isomap
t0 = time()
Y5 = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(256)
plt.scatter(Y5[:, 0], Y5[:, 1], c=Y5[:,0], cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y5)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)

# #MDS
# t0 = time()
# mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# Y = mds.fit_transform(X)
# t1 = time()
# print("MDS: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(256)
# plt.scatter(Y[:, 0], Y[:, 1])
# plt.title("MDS (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.tight_layout()

#spectral embedding
t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y6 = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y6[:, 0], Y6[:, 1], c=Y6[:,0], cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y6)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)

#t-SNE
t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y71 = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y71[:, 0], Y71[:, 1], c=Y71[:,0], cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y71)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)


plt.show()


# In[52]:


from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold

Axes3D

n_points = 1000
X = data_flatten_prot
n_neighbors = 400
n_components = 2

fig = plt.figure(figsize=(15,8))

#standard
t0 = time()
Y12 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'standard').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('standard', t1 - t0))

ax = fig.add_subplot(251)
plt.scatter(Y12[:, 0], Y12[:, 1], c=Y12[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#ltsa
t0 = time()
Y22 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'ltsa').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('ltsa', t1 - t0))

ax = fig.add_subplot(252)
plt.scatter(Y22[:, 0], Y22[:, 1], c=Y22[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LTSA', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#hessian
t0 = time()
Y32 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'hessian').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('hessian', t1 - t0))

ax = fig.add_subplot(253)
plt.scatter(Y32[:, 0], Y32[:, 1], c=Y32[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('Hessian LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#modified LLE
t0 = time()
Y42 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'modified').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('modified', t1 - t0))

ax = fig.add_subplot(254)
plt.scatter(Y42[:, 0], Y42[:, 1], c=Y42[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('Modified LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#isomap
t0 = time()
Y52 = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(256)
plt.scatter(Y52[:, 0], Y52[:, 1], c=Y52[:,0], cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

# #MDS
# t0 = time()
# mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# Y = mds.fit_transform(X)
# t1 = time()
# print("MDS: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(256)
# plt.scatter(Y[:, 0], Y[:, 1])
# plt.title("MDS (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
#plt.tight_layout()

#spectral embedding
t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y62 = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y62[:, 0], Y62[:, 1], c=Y62[:,0], cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#t-SNE
t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y72 = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y72[:, 0], Y72[:, 1], c=Y72[:,0], cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

plt.show()


# In[53]:


fig = plt.figure(figsize=(15,8))

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y12)
ax = fig.add_subplot(251)
plt.scatter(Y12[:, 0], Y12[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y22)
ax = fig.add_subplot(252)
plt.scatter(Y22[:, 0], Y22[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y32)
ax = fig.add_subplot(253)
plt.scatter(Y32[:, 0], Y32[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y42)
ax = fig.add_subplot(254)
plt.scatter(Y42[:, 0], Y42[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y52)
ax = fig.add_subplot(256)
plt.scatter(Y52[:, 0], Y52[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y62)
ax = fig.add_subplot(257)
plt.scatter(Y62[:, 0], Y62[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=1, random_state=0)
clusters = kmeans.fit_predict(Y72)
ax = fig.add_subplot(258)
plt.scatter(Y72[:, 0], Y72[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()


# In[54]:


from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold

Axes3D

n_points = 1000
X = data_flatten_lig_prot
n_neighbors = 400
n_components = 2

fig = plt.figure(figsize=(15,8))

#standard
t0 = time()
Y13 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'standard').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('standard', t1 - t0))

ax = fig.add_subplot(251)
plt.scatter(Y13[:, 0], Y13[:, 1], c=Y13[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#ltsa
t0 = time()
Y23 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'ltsa').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('ltsa', t1 - t0))

ax = fig.add_subplot(252)
plt.scatter(Y23[:, 0], Y23[:, 1], c=Y23[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('LTSA', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(label[mask])[0]
    
# print(accuracy_score(label, labels))

#hessian
t0 = time()
Y33 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'hessian').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('hessian', t1 - t0))

ax = fig.add_subplot(253)
plt.scatter(Y33[:, 0], Y33[:, 1], c=Y33[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('Hessian LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#modified LLE
t0 = time()
Y43 = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method = 'modified').fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ('modified', t1 - t0))

ax = fig.add_subplot(254)
plt.scatter(Y43[:, 0], Y43[:, 1], c=Y43[:,0], cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ('Modified LLE', t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#isomap
t0 = time()
Y53 = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(256)
plt.scatter(Y53[:, 0], Y53[:, 1], c=Y53[:,0], cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

# # #MDS
# # t0 = time()
# # mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# # Y = mds.fit_transform(X)
# # t1 = time()
# # print("MDS: %.2g sec" % (t1 - t0))
# # ax = fig.add_subplot(256)
# # plt.scatter(Y[:, 0], Y[:, 1])
# # plt.title("MDS (%.2g sec)" % (t1 - t0))
# # ax.xaxis.set_major_formatter(NullFormatter())
# # ax.yaxis.set_major_formatter(NullFormatter())
# plt.tight_layout()

#spectral embedding
t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y63 = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y63[:, 0], Y63[:, 1], c=Y63[:,0], cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

#t-SNE
t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y73 = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y73[:, 0], Y73[:, 1], c=Y73[:,0], cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()

plt.show()


# In[55]:


fig = plt.figure(figsize=(15,8))

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y13)
ax = fig.add_subplot(251)
plt.scatter(Y13[:, 0], Y13[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y23)
ax = fig.add_subplot(252)
plt.scatter(Y23[:, 0], Y23[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y33)
ax = fig.add_subplot(253)
plt.scatter(Y33[:, 0], Y33[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y43)
ax = fig.add_subplot(254)
plt.scatter(Y43[:, 0], Y43[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y53)
ax = fig.add_subplot(256)
plt.scatter(Y53[:, 0], Y53[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y63)
ax = fig.add_subplot(257)
plt.scatter(Y63[:, 0], Y63[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(Y73)
ax = fig.add_subplot(258)
plt.scatter(Y73[:, 0], Y73[:, 1], c=clusters,
            s=50, cmap=plt.cm.Spectral)
#label = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=30, alpha=0.5)
plt.tight_layout()


# In[56]:


mod_class = pd.read_csv("mod_class.csv")
mod_class.head()


# In[ ]:




