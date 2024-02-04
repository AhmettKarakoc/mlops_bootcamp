import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.hist(df["gdp_per_capita"])
ax2.hist(gdp_trnsform)
ax1.set_title("Gdp_Per_Capita Before Log Transform")
ax2.set_title("Gdp_Per_Capita After Log Transformed")
ax1.set_ylabel("GDP")

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.hist(df["total_deaths"])
ax2.hist(total_deaths_trnsform)
ax1.set_title("Total_Deaths Before Log Transform")
ax2.set_title("Total_Deaths After Log Transformed")
ax1.set_ylabel("Total Deaths")

f = plt.figure(figsize=(6,6))
ax = plt.axes()

labels = ['Ridge', 'Lasso', 'ElasticNet']

models = [ridgecv, lassocv, elasticnetcv]

for mod, lab in zip(models, labels):
    ax.plot(y_test, mod.predict(x_test), 
             marker='o', ls='', ms=3.0, label=lab, alpha=0.9)


leg = plt.legend(frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.0)

ax.set(xlabel='Actual ', 
       ylabel='Predicted ', 
       title='Linear Regression Results')