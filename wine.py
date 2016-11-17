# The Wine Equation (regression on prices of Bordeaux vintages)
import libwine as lw
import numpy as np
import pandas as pd

# Read and prepare the data
wine = pd.read_csv("wine.csv")
wine['Ones'] = 1 # For the intercept term
for col in ['AGST', 'WinterRain', 'HarvestRain', 'Age']:
    wine[col] = lw.normalize(wine[col])

# Gradient descent!
t = lw.descent(wine.Price.values, wine[['Ones', 'AGST']].values, alpha=1e-2)
print(t)
print(lw.r2(wine.Price.values, t, wine[['Ones', 'AGST']].values))

# Scatter plot of wine price against AGST
import matplotlib.pyplot as plt
x = np.array([wine.AGST.min()-1, wine.AGST.max()+1]) # just for plotting
plt.ion()
plt.figure()
plt.plot(wine.AGST, wine.Price, 'ro')
plt.xlabel("Average Growing Season Temperature")
plt.ylabel("log(Price)")
plt.grid("on")
plt.plot(x, t[0] + t[1] * x, 'b-', label="regression line")
plt.xlim(x)
plt.title("The Wine Equation")
plt.savefig("wine.png")

# A more complex model
t = lw.descent(wine.Price.values, wine[['Ones', 'AGST', 'WinterRain', 'HarvestRain', 'Age']].values, alpha=1e-2)
print(t)
print(lw.r2(wine.Price.values, t, wine[['Ones', 'AGST', 'WinterRain', 'HarvestRain', 'Age']].values))
