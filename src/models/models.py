import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Mute the sklearn warning about regularization
import warnings
warnings.filterwarnings('ignore', module='sklearn')

df = pd.get_dummies(df, columns=["continents"], drop_first=True)

df = df.drop("date", axis=1)

#Train Models
y = df.gdp_per_capita
x = df.drop("gdp_per_capita", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state = 157)

kf = KFold(shuffle = True , random_state= 157, n_splits= 3 )
#LinearRegression
ss = StandardScaler()
lr = LinearRegression()

x_train_ss = ss.fit_transform(x_train)
lr.fit(x_train_ss, y_train)
x_test_ss = ss.transform(x_test)
y_pred = lr.predict(x_test_ss)
score = r2_score(y_test.values, y_pred)

#with pipeline
estimator=Pipeline([("scaler",ss),("linear_reg",lr)])
prediction = cross_val_predict(estimator, x_train_ss,y_train, cv=kf)
linear_score = r2_score(y_train, prediction)

score, linear_score

#Ridge regression
pf = PolynomialFeatures(degree=2)
alphas = np.geomspace(4,20,20)

score=[]
for alpha in alphas:
    ridge = Ridge(alpha = alpha, max_iter=10000)
    
    estimator = Pipeline([("scaler", ss),("polynomial_features", pf),("ridge_reg",ridge)])
    prediction_r = cross_val_predict(estimator, x_train_ss, y_train,cv=kf) 
    scores = r2_score(y_train,prediction_r)
    score.append(scores)
score
plt.plot(alphas, score, "-o")
plt.title("Ridge Regression")
plt.xlabel("$\\alpha$")
plt.ylabel("$R^2$")

best_estimator= Pipeline([("scaler",ss),("polynomial_reg",pf),("ridge_reg",Ridge(alpha=0.4))])
best_estimator.fit(x_train_ss,y_train)
ridge_score = best_estimator.score(x_train_ss,y_train)

#lasso regression
pf = PolynomialFeatures(degree=3)

score = []
alphas = np.geomspace(0.06,6.0,20)
for alpha in alphas:
    lasso = Lasso(alpha = alpha, max_iter=10000)
    estimator = Pipeline([("scaler",ss),("polynomial_features",pf),("lasso",lasso)])
    prediction_l = cross_val_predict(estimator,x_train_ss, y_train, cv=kf )
    scores = r2_score(y_train, prediction_l)
    score.append(scores)
plt.plot(alphas,score, "-o")
plt.title("Lasso Regression")
plt.xlabel("$\\alpha$")
plt.ylabel("$R^2$")

best_estimator = Pipeline([("scaler",ss),("polynomial_features",pf),("lasso_reg",lasso)])
best_estimator.fit(x_train_ss,y_train)
lasso_score = best_estimator.score(x_train_ss, y_train)

pd.DataFrame([[linear_score,ridge_score,lasso_score]],columns=["Linear","Ridge","Lasso"], index=["score"])

def rmse (ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue,ypredicted))
#linear
linear_regs = LinearRegression().fit(x_train,y_train)
linear_rmse = rmse(y_test, linear_regs.predict(x_test))
#Ridge
alphas0 = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]
ridgecv = RidgeCV(alphas = alphas0, cv=4).fit(x_train,y_train)
ridgecv_rmse = rmse(y_test, ridgecv.predict(x_test))
#Lasso
alphas1 = np.array([1e-5, 5e-5, 0.0001, 0.0005])
lassocv = LassoCV(alphas=alphas1,max_iter = 10000, cv=4).fit(x_train,y_train)
lassocv_rmse = rmse(y_test, lassocv.predict(x_test))
#ElasticnetCV 
l1_ratios = np.linspace(0.1,0.9,9)
elasticnetcv = ElasticNetCV(alphas = alphas1, l1_ratio=l1_ratios).fit(x_test,y_test)
elasticnetcv_rmse = rmse(y_test, elasticnetcv.predict(x_test))
rmse_vals = [linear_rmse, ridgecv_rmse, lassocv_rmse, elasticnetcv_rmse]
labels = ['Linear', 'Lasso', 'Ridge' 'ElasticNet']

rmse_df = pd.DataFrame([[linear_rmse, ridgecv_rmse, lassocv_rmse, elasticnetcv_rmse]],columns=['Linear', 'Lasso', 'Ridge', 'ElasticNet'], index=['rmse'])
rmse_df
