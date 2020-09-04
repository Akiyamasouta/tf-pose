import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parts = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',  5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8:'RHip', 9: 'RKnee',
         10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'}

data = pd.read_csv("test0.csv")
fig = plt.figure()

#ax = fig.add_subplot(1,2,1)
#ax.plot(data['t'], data['x'], '.')
#ax.set_xlabel('time', fontSize=14)
#ax.set_ylabel('x', fontSize=14)

ax = fig.add_subplot(1,2,2)
ax.plot(data[data['parts']=='Nose']['t'], data[data['parts']=='Nose']['y'], '.')
ax.set_xlabel('time', fontSize=14)
ax.set_ylabel('y', fontSize=14)

plt.show()


