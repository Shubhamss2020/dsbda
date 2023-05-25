# Assigment No. 2 - Data Wrangling II
# • Program by - Sir, Witten By your friends ^_^
# agli baar padhlo bhosadiwalo

#interquartile range (IQR) method.
def RemoveOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high = Q3 + 1.5 * IQR
    low = Q1 - 1.5 * IQR

    df = df[(df[var] >= low) & (df[var] <= high)]
    print('Outliers removed in', var)
    return df

def DisplayOutliers(df, message):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(message)
    sns.boxplot(data=df, x='raisedhands', ax=axes[0, 0])
    sns.boxplot(data=df, x='VisITedResources', ax=axes[0, 1])
    sns.boxplot(data=df, x='AnnouncementsView', ax=axes[1, 0])
    sns.boxplot(data=df, x='Discussion', ax=axes[1, 1])
    fig.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read dataset
df = pd.read_csv('dataset.csv')
print('Boston dataset loaded.')

#Display information about dataset
print('Information of dataset:\n', df.info)
print('Shape of dataset (row x column):', df.shape)
print('Column names: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatypes of attributes:\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

# Display statistical information of dataset
print("Statistical information of Numerical Columns:\n", df.describe())

# Display null values
print("Total Number of Null Values in Dataset:", df.isna().sum())


#Handling outliers
DisplayOutliers(df, 'Before removing outliers')
df = RemoveOutlier(df, 'raisedhands')
df = RemoveOutlier(df, 'VisITedResources')
df = RemoveOutlier(df, 'AnnouncementsView')
df = RemoveOutlier(df, 'Discussion')
DisplayOutliers(df, 'After removing outliers')

# Conversion of Categorical to Quantitative (Encoding)
df['gender'] = df['gender'].astype('category')
df['gender'] = df['gender'].cat.codes

print('Data types of gender after label encoding:', df.dtypes['gender'])
print('Gender Values:', df['gender'].unique())


sns.boxplot(data=df, x='gender', y='raisedhands', hue='gender')
plt.title('Boxplot with 2 variables gender and raisedhands')
plt.show()

sns.boxplot(data=df, x='NationalITy', y='Discussion', hue='gender')
plt.title('Boxplot with 3 variables Gender, Nationality and Discussion')
plt.show

print('Relationship between variables using Scatterplot:')
sns.scatterplot(data=df, x='raisedhands', y='VisITedResources')
plt.title('Scatterplot for raisedhands and VisITedResources')
plt.show()  

#OBJ 3 :

# Square root transformation - converting non-linear relation
df['sqrt_AnnouncementsView'] = np.sqrt(df['AnnouncementsView'])

# How change is happened
print("\n\n\\n")
print(df['AnnouncementsView'])
print("\n\n\\n")
print(df['sqrt_AnnouncementsView'])
print("\n\n\\n")
df['sqrt_AnnouncementsView']=df['sqrt_AnnouncementsView'].astype('int8')
print(df['sqrt_AnnouncementsView'])
