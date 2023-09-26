import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from mplot3d_dragger import Dragger3D


csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(csv_path, header=None, sep='\s+')

df.columns = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD'
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT',
    'MEDV'
]

df.head()

# Understanding the dataset

# Linear regression is based on correlations so first I plot a pair plot with Seaborn to visualize correlations between
# our variables

sns.set(style='whitegrid', context='notebook')
cols = ['DIS', 'INDUS', 'CRIM', 'RM', 'MEDV']
sns.pairplot(df[cols], height=2.5)
plt.show()

# After taking a look at correlations between variables, it is important to see this numerically, so we plot a
# correlation matrix in order to do so

correlation_matrix = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
sns.heatmap(correlation_matrix, cbar=True, annot=True, yticklabels=cols, xticklabels=cols)

# Time to create the model with scikit-learn. Initially I will perform one-variable prediction

X = df['RM'].values.reshape(-1, 1)
y = df['MEDV'].values.reshape(-1, 1)

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

slr = LinearRegression()
slr.fit(X_std, y_std)

plt.scatter(X_std, y_std)
plt.plot(X_std, slr.predict(X_std), color='R')
plt.ylabel("Median home price [MEDV]")
plt.xlabel("Average number of rooms [RM]")

# Generating model prediction

num_rooms = 5
num_rooms_std = num_rooms.transform(np.array([num_rooms]).reshape(-1, 1))
print(f'A 5 room house in Boston costs ${sc_y.inverse_transform(slr.predict(num_rooms_std))} USD')

# Multi-variable prediction

X = df[['RM', 'INDUS']].values
y = df['MEDV'].values.reshape(-1, 1)

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

slr = LinearRegression()
slr.fit(X_std, y_std)

x1_range = np.arange(df['RM'].min(), df['RM'].max())
x2_range = np.arange(df['INDUS'].min(), df['INDUS'].max())

X1, X2 = np.meshgrid(x1_range, x2_range)

plane = pd.DataFrame({'RM': X1.ravel(), 'INDUS': X2.ravel()})
prediction = slr.predict(plane).reshape(X1.shape)
prediction = sc_y.inverse_transform(prediction)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X1, X2, prediction, alpha=0.4)

ax.scatter3D(df['RM'], df['INDUS'], df['MEDV'], color='R', marker='.')
ax.view_init(elev=10, azim=5)
dr = Dragger3D(ax, real_time=True)
plt.show()

# Model prediction

num_rooms_std = sc_x.transform(np.array([5.0]).reshape(-1, 1))
price_std = slr.predict(num_rooms_std)
print(f'Price: ${sc_y.inverse_transform(price_std)} USD')
