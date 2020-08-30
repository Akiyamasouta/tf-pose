import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parts = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',  5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8:'RHip', 9: 'RKnee',
         10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'}

df = pd.read_csv("test0.csv")
df.set_index("x", inplace=True)
pt = pd.pivot_table(df, index="x", columns=["parts"], values="y")
pt.plot.line(legend=True)
plt.show()


