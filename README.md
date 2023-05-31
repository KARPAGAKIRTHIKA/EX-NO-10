# Ex:10 Data Science Process on Complex Dataset
# AIM:
To Perform Data Science Process on a complex dataset and save the data to a file

# ALGORITHM:
STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Generation/Feature Selection Techniques on the data set

STEP 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE:
Data Cleaning Process:
```
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("Iris.csv")

df.head(10)

df.info()

df.describe()

df.isnull().sum()

Handling Outliers:
q1 = df['SepalLengthCm'].quantile(0.25)

q3 = df['SepalLengthCm'].quantile(0.75)

IQR = q3 - q1

print("First quantile:", q1, " Third quantile:", q3, " IQR:", IQR, "\n")

lower = q1 - 1.5 * IQR

upper = q3 + 1.5 * IQR

outliers = df[(df['SepalLengthCm'] >= lower) & (df['SepalLengthCm'] <= upper)]

from scipy.stats import zscore

z = outliers[(zscore(outliers['SepalLengthCm']) < 3)]

print("Cleaned Data: \n")

print(z)

EDA Techniques:
df.skew()

df.kurtosis()

sns.boxplot(x="SepalLengthCm",data=df)

sns.boxplot(x="SepalWidthCm",data=df)

sns.countplot(x="Species",data=df)

sns.distplot(df["PetalWidthCm"])

sns.distplot(df["PetalLengthCm"])

sns.histplot(df["SepalLengthCm"])

sns.histplot(df["PetalWidthCm"])

sns.scatterplot(x=df['SepalLengthCm'],y=df['SepalWidthCm'])

import matplotlib.pyplot as plt

states=df.loc[:,["Species","SepalLengthCm"]]

states=states.groupby(by=["Species"]).sum().sort_values(by="SepalLengthCm")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="SepalLengthCm",data=states)

plt.xlabel=("Species")

plt.ylabel=("SepalLengthCm")

plt.show()

import matplotlib.pyplot as plt

states=df.loc[:,["Species","PetalWidthCm"]]

states=states.groupby(by=["Species"]).sum().sort_values(by="PetalWidthCm")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="PetalWidthCm",data=states)

plt.xlabel=("Species")

plt.ylabel=("PetalWidthCm")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

Feature Generation:
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

le=LabelEncoder()

df['Id']=le.fit_transform(df['SepalLengthCm'])

df

S=['Iris-setosa','Iris-virginica','Iris-versicolor']

enc=OrdinalEncoder(categories=[S])

enc.fit_transform(df[['Species']])

df['SP1']=enc.fit_transform(df[['Species']])

df

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from google.colab import files

uploaded = files.upload()

from sklearn.preprocessing import OneHotEncoder

df1 = pd.read_csv("Iris.csv")

ohe=OneHotEncoder(sparse=False)

enc=pd.DataFrame(ohe.fit_transform(df1[['Species']]))

df1=pd.concat([df1,enc],axis=1)

df1
```
Feature Transformation:
```
import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df1['SepalLengthCm'],fit=True,line='45')

plt.show()

sm.qqplot(df1['SepalWidthCm'],fit=True,line='45')

plt.show()

import numpy as np

from sklearn.preprocessing import PowerTransformer

transformer=PowerTransformer("yeo-johnson")

df1['PetalLengthCm']=pd.DataFrame(transformer.fit_transform(df1[['PetalLengthCm']]))

sm.qqplot(df1['PetalLengthCm'],line='45')

plt.show()

transformer=PowerTransformer("yeo-johnson")

df1['PetalWidthCm']=pd.DataFrame(transformer.fit_transform(df1[['PetalWidthCm']]))

sm.qqplot(df1['PetalWidthCm'],line='45')

plt.show()

qt=QuantileTransformer(output_distribution='normal')

df1['SepalWidthCm']=pd.DataFrame(qt.fit_transform(df1[['SepalWidthCm']]))

sm.qqplot(df1['SepalWidthCm'],line='45')

plt.show()

Data Visua/lization:
sns.barplot(x="Species",y="SepalLengthCm",data=df1)

plt.xticks(rotation = 90)

plt.show()

sns.lineplot(x="PetalLengthCm",y="PetalWidthCm",data=df1,hue="Species",style="Species")

sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",hue="Species",data=df1)

sns.histplot(data=df1, x="SepalLengthCm", hue="Species", element="step", stat="density")

sns.relplot(data=df1,x=df1["PetalWidthCm"],y=df1["PetalLengthCm"],hue="Species")
```
# OUTPUT:
# Data Cleaning Process:

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/a44a0136-f0b9-4262-87be-2709357a451e)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/100303c5-eae6-47d8-a095-5356018bc308)


# Handling Outliers:

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/9fa5f851-15cb-4f5f-ad31-79257905e7f6)


# EDA Techniques:

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/1e6c019a-6167-493c-acc5-e687259597ea)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/490c56eb-f02f-4c07-985b-d5f2947ef6a2)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/7edd191c-549b-4b6c-900f-008643190a6d)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/863502b0-4ecf-44d6-b2c1-df0c1b5b888f)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/793d455a-1fd5-4b10-a5bd-e0cd828a55c7)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/1bafc75f-2f99-41c7-802b-f59100ded57f)


# Feature Generation:

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/07a49c70-de74-4d79-9ec1-906edf717f5e)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/9d1ac82c-9208-4135-83a2-283f6dfa9d98)


# Feature Transformation:

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/04485a64-a39d-460f-9957-63cd580d9686)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/1fe767c9-d080-49b8-8a06-64fffa324f32)


# Data Visua/lization:


![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/cb0c818e-7991-4417-92fe-18977179f8a0)

![image](https://github.com/KARPAGAKIRTHIKA/EX-NO-10/assets/103020162/94f7d722-e419-4f8c-bb80-c9319cc9f564)


# RESULT:
Thus the Data Science Process on Complex Dataset were performed and output was verified successfully.

