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

df = pd.read_csv("raw_data.csv")
df.head()

df.isnull().sum()

df = df.drop(["iso_code","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"], axis=1)

df.dtypes

df['total_cases'].fillna(df['total_cases'].mean(), inplace=True)
df['total_deaths'].fillna(df['total_deaths'].mean(), inplace=True)
df['stringency_index'].fillna(df['stringency_index'].mean(), inplace=True)
df['gdp_per_capita'].fillna(df['gdp_per_capita'].mean(), inplace=True)
df['human_development_index'].fillna(df['human_development_index'].mean(), inplace=True)
df.isnull().sum()

df = df.rename(columns={"human_development_index":"hdi"})

gdp_trnsform = df['gdp_per_capita'].apply(lambda x: np.log(x+1 ))
total_deaths_trnsform = df['total_deaths'].apply(lambda x: np.log(x+1))

df["gpd_per_capita"]= gdp_trnsform
df["total_deaths"] = total_deaths_trnsform
df.head()

asia_countries = [
    'Afghanistan', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'United Arab Emirates', 'Brunei', 'China',
    'East Timor', 'Indonesia', 'Armenia', 'Philippines', 'Palestine', 'South Korea', 'Georgia', 'India', 'Iraq',
    'Iran', 'Israel', 'Japan', 'Cambodia', 'Qatar', 'Kazakhstan', 'Kyrgyzstan', 'North Korea', 'Malaysia', 'Nepal',
    'Uzbekistan', 'Pakistan', 'Russia', 'Singapore', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Turkey',
    'Turkmenistan', 'Jordan', 'Vietnam', 'Yemen']

africa_countries = [
    'Angola', 'Western Sahara', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Algeria', 'Djibouti', 'Chad',
    'Congo DC', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Morocco', 'Ivory Coast', 'Gabon', 'Gambia', 'Ghana',
    'Guinea', 'Guinea-Bissau', 'Republic of South Africa', 'Cameroon', 'Cape Verde', 'Kenya', 'Comoros', 
    'Republic of the Congo', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritius', 'Mauritania',
    'Mozambique', 'Egypt', 'Namibia', 'Niger', 'Nigeria', 'Central African Republic', 'Rwanda', 'Sao Tome and Principe',
    'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'Sudan', 'Eswatini', 'Tanzania', 'Togo', 'Tunisia', 'Uganda',
    'Zambia', 'Zimbabwe']

europe_countries = [
    'Andorra', 'Germany', 'Albania', 'Azerbaijan', 'Austria', 'Belgium', 'Belarus', 'United Kingdom', 
    'Bosnia and Herzegovina', 'Bulgaria', 'Georgia', 'Czechia', 'Denmark', 'Estonia', 'Faroe Islands', 
    'Finland', 'France', 'Netherlands', 'Croatia', 'Ireland', 'Spain', 'Sweden', 'Switzerland', 'Italy', 
    'Iceland', 'Montenegro', 'Kosovo', 'Turkish Republic of Northern Cyprus', 'Latvia', 'Liechtenstein', 
    'Lithuania', 'Luxembourg', 'Hungary', 'North Macedonia', 'Malta', 'Moldova', 'Monaco', 'Norway', 
    'Poland', 'Portugal', 'Romania', 'Rusya', 'San Marino', 'Slovakia', 'Slovenia', 'Sırbistan', 'Türkiye', 
    'Ukraina', 'Greece']

north_america_countries = [
    'US Virgin Islands', 'United States', 'Antigua and Barbuda', 'Aruba', 'Bahama', 'Barbados', 'Belize', 'Bermuda',
    'Cayman Islands', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Greenland', 'Guadeloupe', 'Guatemala',
    'Haiti', 'Netherlands Antilles', 'Honduras', 'Jamaica', 'Canada', 'Costa Rica', 'Cuba', 'Martinique', 'Mexico',
    'Montserrat', 'Nicaragua', 'Panama', 'Puerto Rico', 'Saint Kitts ve Nevis', 'Saint Lucia', 'Saint Vincent ve Granada',
    'Saint-Pierre ve Miquelon', 'Trinidad ve Tobago', 'Turks ve Caicos Islands']

south_america_countries = [
    'Argentina', 'Bolivia', 'Brazil', 'French Guiana', 'Guyana', 'Colombia', 'Ecuador', 'Paraguay', 'Surinam',
    'Uruguay', 'Venezuela', 'Şili', 'Peru']

oceania_countries = [
    'New Caledonia', 'Australia', 'Tonga', 'Vanuatu', 'Nauru', 'New Zealand', 'Fiji', 'Tuvalu', 'Samoa', 
    'Solomon Islands', 'Kiribati', 'Papua New Guinea']

for country in asia_countries:
    df["location"] = df["location"].str.replace(country, "Asia")

for country in africa_countries:
    df["location"] = df["location"].str.replace(country, "Africa")
    
for country in europe_countries:
    df["location"] = df["location"].str.replace(country, "Europe")
    
for country in north_america_countries:
    df["location"] = df["location"].str.replace(country, "North_America")
    
for country in south_america_countries:
    df["location"] = df["location"].str.replace(country, "South_America") 
    
for country in oceania_countries:
    df["location"] = df["location"].str.replace(country, "Oceania")

df.rename(columns={'location': 'continents'}, inplace=True)