```python
#Importing Data 
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
data=pd.read_csv("/content/drive/Shareddrives/Python group 5/heart.csv", na_values=["unknown\"\""])
data

#Data Analysis
data.sample(5)
data.info()
data.describe()

##Age
sns.displot(data['Age'],kde=True)
sns.violinplot(y=data["Age"],x=data["HeartDisease"])

##Sex
hist=px.histogram(data['Sex'])
hist.show()
data.groupby("Sex").size().plot(kind="pie",autopct= '%.2f')

##Chest Pain Type
hist=px.histogram(data['ChestPainType'])
hist.show()
data.groupby("ChestPainType").size().plot(kind="pie",autopct= '%.2f')

##Resting BP
hist=px.histogram(data['RestingBP'])
hist.show()

##Cholesterol
hist=px.histogram(data['Cholesterol'])
hist.show()

##Fasting BS
hist=px.histogram(data['FastingBS'])
hist.show()

##Resting ECG
data.groupby('RestingECG').size().plot(kind="pie")

##Max HR
hist=px.histogram(data['MaxHR'])
hist.show()

##Exercise Angina
hist=px.histogram(data['ExerciseAngina'])
hist.show()

##Old Peak
hist=px.histogram(data['Oldpeak'])
hist.show()

##ST Slope
hist=px.histogram(data['ST_Slope'])
hist.show()

##Overall Heart Analysis
a=sns.pairplot(data, hue='HeartDisease')
target = ['HeartDisease']

num_attribs = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
cat_nom_attribs = ['ChestPainType', 'RestingECG', 'ST_Slope']
cat_bin_attribs = ['Sex', 'FastingBS', 'ExerciseAngina']
cat_attribs = cat_nom_attribs + cat_bin_attribs
ncol = 3
nrow = int(np.ceil(len(num_attribs)/ncol))

fig, axs = plt.subplots(nrow, ncol, figsize=(10, 5), facecolor=None)   

i = 1
for col in num_attribs:
    plt.subplot(nrow, ncol, i)
    ax = sns.histplot(data=data, x=col, hue=target[0], multiple="stack", palette='colorblind')
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    sns.despine(right=True)
    sns.despine(offset=0, trim=False)
    i+=1

fig.delaxes(axs[nrow-1, ncol-1])
plt.suptitle('Distribution of Numerical Features', fontsize = 14);
plt.tight_layout()   
ncol = 3
nrow = int(np.ceil(len(num_attribs)/ncol))

f, axes = plt.subplots(nrow, ncol, figsize=(8,6))

for name, ax in zip(num_attribs, axes.flatten()):
    sns.boxplot(y=name, x= "HeartDisease", data=data, orient='v', ax=ax)

f.delaxes(axes[nrow-1, ncol-1])
plt.suptitle('Box-and-whisker plot', fontsize = 14);
plt.tight_layout()    
**1.Categorical features** - 7 categorical features, including the target feature - 3 nominal: ChestPainType, RestingECG, ST_Slope
- 4 binary features: Sex, FastingBS, ExerciseAngina, HeartDisease (target)
- HeartDisease (target variable): 55% of the sample have heart disease. Therefore, feature values that occur in smaller subsamples can be ruled out as main diagnostics of the disease. We hightlight frequent feature values in bold. - **Sex:** Male comprise 79% of the sample. Number of females is 193, relatively low. - **ChestPainType** has 4 values, among which **ASY** (asymptomatic) occurs most frequently (54%), followed by similar values (22 and 19%) for NAP and ATA. TA occurs only in 5% - FastingBS has values of 1 (elavated blood sugar) in 23% of the sample - RestingECG: 60% of the sample has normal ECG, the rest is equally divided between ST and LVH types - **ExerciseAngina**: 40% of the sample have exercise-induced angina (**Y**) and 60% do not. - **ST_Slope**: 50% of the sample has **Flat** slope of the peak exercise ST segment and 47% - upslopping (**Up**) **2. Numerical features** - 5 numerical interval features: - Age, RestingBP, Cholesterol, MaxHR, Oldpeak - Most of the sample is in the middle age. The mean age is 53. 25% are younger than 47 and 25% are older than 60. - OldPeak minimum value of -2.6 is far from the mean of 0.88. Are the data points with the lowest values outliers? - MaxHR shows large spread of values. This could be related to the known age-MaxHR dependence, but could also be an indicator of the heart disease. - There is no NAN values in the dataset. Cholesterol feature has 172 zero values and and RestingBP features has one zero value. Since this is not medically possible, we conclude that these are missing values.

###Correlations between features
for attr in cat_attribs:
    display(data[[attr, 'HeartDisease']].groupby(attr, as_index=False).mean().sort_values(by='HeartDisease', ascending=False))
    **Conclusions from pivoting feature analysis** We find that *ChestPainType*, *ExerciseAngina*, *ST_Slope*, and *Sex* features are more important for diagnosis. - *Sex* feature shows correlation with the heart disease. Males have a higher fraction of heart disease than females (63% vs 26%). Although the sample of female is relatively small. - *ChestPainType:* 79% of those who show Asymptomatic ChestPainType have heart disease. Less than a half, 43% and 35%, respectively, TA and NAP chest pain types also have heart disease. Is chest pain not a good individual indicator of the heart disease, if so many cases are asymptomatic? - *ExerciseAngina:* 85% of those who exercise angina have heart disease, in contrast to 35% who don't exercise angina - *ST_Slope:* 82% of flat ST Slope also have heart disease, in contrast to 20% who have upward slope - *FastingBS:* There is a correlation between FastingBS=1 and heart disease (79% of the sample with elavated fasting blood sugar have heart disease). A bit less than a half of the sample without elevated also have heart disease. Given that only 23% of the sample have FastingBS=1, we do not expect this variable to be an important diagnostic. - *RestingECG* feature does not show strong correlation as other features.
    grid = sns.FacetGrid(data, col='ST_Slope', height=3., aspect=1.2)
grid.map(sns.pointplot, 'ChestPainType' , 'HeartDisease', 'Sex', 
         hue_order=['M', 'F'], order=['ASY', 'ATA', 'NAP', 'TA'], palette='colorblind')
grid.add_legend()

grid = sns.FacetGrid(data, col='ExerciseAngina', height=3., aspect=1.2)
grid.map(sns.pointplot, 'ChestPainType' , 'HeartDisease', 'Sex', 
         hue_order=['M', 'F'], order=['ASY', 'ATA', 'NAP', 'TA'], palette='colorblind')
grid.add_legend()
grid = sns.FacetGrid(data, col='ExerciseAngina', height=3., aspect=1.2)
grid.map(sns.pointplot, 'RestingECG' , 'HeartDisease', 'Sex', 
         hue_order=['M', 'F'], order=['Normal', 'LVH', 'ST'], palette='colorblind')
grid.add_legend()
We perform an additional test of our assumption that data for both sexes can be used for diagnostic together. We calculate means for numerical features MaxHR and OldPeak for the sample of men and women with and without heart condition and find that the differences between these values of about 10%.
print('MaxHR values for F, M:')
print('HeartDisease=0', round(data.loc[ (data["Sex"]=="F") & (data["HeartDisease"]==0) ]['MaxHR'].mean(),2), 
      round(data.loc[ (data["Sex"]=="M") & (data["HeartDisease"]==0) ]['MaxHR'].mean(),2))
print('HeartDisease=1', round(data.loc[ (data["Sex"]=="F") & (data["HeartDisease"]==1) ]['MaxHR'].mean(),2),
      round(data.loc[ (data["Sex"]=="M") & (data["HeartDisease"]==1) ]['MaxHR'].mean(),2))

print('Oldpeak values for F, M:')
print('HeartDisease=0', round(data.loc[ (data["Sex"]=="F") & (data["HeartDisease"]==0) ]['Oldpeak'].mean(),2),
      round(data.loc[ (data["Sex"]=="M") & (data["HeartDisease"]==0) ]['Oldpeak'].mean(),2))
print('HeartDisease=1', round(data.loc[ (data["Sex"]=="F") & (data["HeartDisease"]==1) ]['Oldpeak'].mean(),2), 
      round(data.loc[ (data["Sex"]=="M") & (data["HeartDisease"]==1) ]['Oldpeak'].mean(),2))
      **Conclusions from association analysis between categorical features** - We confirm our findings from pivoting features that *ChestPainType*, *ExerciseAngina*, *ST_Slope* features are more important heart diagnostics. Among them, *ST_Slope* has the strongest association with heart condition (Cramér's V=0.69), followed by *ChestPainType* (V=0.54) and *ExerciseAngina* (V=0.49). In contrast, *Sex* feature has the lowest association with heart condition (V=0.3) among these four attributes. - We now can answer on how important categorical features relate to each other. *ExerciseAngina* has the strongest associations with *ST_Slope* and *ChestPainType* (V=0.45). The coefficient for the *ST_Slope* and *ChestPainType* pair implies a lower association (V=0.29).
      data.corr()
      def cramers_v(x, y): 
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# calculate the correlation coefficients
data_ = data[cat_attribs+target]
rows= []
for x in data_:
    col = []
    for y in data_ :
        col.append(cramers_v(data_[x], data_[y]) )
    rows.append(col)
    
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = data_.columns, index = data_.columns)

# heatmap plot
mask = np.triu(np.ones_like(df, dtype=bool))
fig, ax = plt.subplots(figsize=(8, 6), facecolor=None)
sns.heatmap(df, cmap=sns.color_palette("husl", as_cmap=True), 
            vmin=0, vmax=1.0, center=0, annot=True, fmt='.2f', 
            square=True, linewidths=.01, cbar_kws={"shrink": 0.8})
ax.set_title("Association between categorical variables (Cramer's V)", fontsize=14)
Conclusions from correlations of numerical features - There are no strong correlations, which we keep all attributes - A moderate anticorrelation with coefficient of -0.38 is found between *MaxHR* and *Age*. It agrees with a well known relationship between maximum heart rate and age. - When we compare correlations with the target variable, we find the strongest relationship for *MaxHR* and *OldPeak* features. Their coefficients of 0.4 and -0.4 imply moderate relationships. - Coefficient for *RestingBP* feature of 0.11 implies almost no correlation. We can consider removing this feature from the machine learning.
corr_matrix = data[num_attribs+target].corr()

# plotting correlation heatmap
dataplot = sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, fmt='.2f')
plt.show()
**Pivoting features** - *Sex* is an important factor, since males have a significantly higher fraction of heart disease than females (63% vs 26%) in our sample. Correlations between other categorical feature values and the target for female sample are similar to those of males. - *ChestPainType:* 79% of those who show Asymptomatic ChestPainType have heart disease. Less than a half, 43% and 35%, respectively, TA and NAP chest pain types also have heart disease. - *ExerciseAngina:* 85% of those who exercise angina have heart disease, in contrast to 35% who don't exercise angina - *ST_Slope:* 82% of flat ST Slope also have heart disease, in contrast to 20% who have upward slope **Associations between categorical features** - We find no strong correlations with the target variable for both categorical and numeric features. It is expected, since otherwise it would be much easier to diagnose heart disease. - *ST_Slope* feature has the higest association with *HeartDisease*, followed by *ChestPainType*, *ExerciseAngina*, and *Sex*. The latter has the lowest association with the target variable among these four attributes. - *ST_Slope* has the strongest association with *ExerciseAngina* feature. *ExerciseAngina* has similarly strong correlation with *ChestPainType*. *ST_Slope* and *ChestPainType* pair shows lower correlation. **Correlations between numerical features** - There are also no strong correlations between numerical features, with the highest correlation coefficient of -0.38 found for MaxHR and Age). - When we compare correlations with the target variable, we find the strongest, albight moderate correlation for *OldPeak* feature and moderate anticorrelation for *MaxHR*. Based on EDA, we expect categorical features **ST_Slope**, **ChestPainType**, **ExerciseAngina**, and **Sex** and numerical features **MaxHR** and **OldPeak** to be important for prediction of heart disease in the following machine learning analysis.
```python
