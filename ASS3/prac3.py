import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('Employee_Salary_Dataset.csv')
print('Boston dataset loaded.')

# Display information about dataset
print('Information of dataset:\n', df.info())
print('Shape of dataset (row x column):', df.shape)
print('Column names:', df.columns)
print('Total elements in dataset:', df.size)
print('Datatypes of attributes:\n', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

# Statistical information of Numerical Columns
columns = ['Experience_Years', 'Age', 'Salary']
for column in columns:
    print('\n')
    print("****************************************  "+column+"  **************************************************")
    print(df[column].describe())
    print('\n')

# Groupwise Statistical Summary
for column in columns:
    print('\n')
    print("****************************************  "+column+"  **************************************************")
    print(df.groupby('Gender')[column].describe())
    print('\n')

# Plotting Groupwise Statistical Information
X = ['min', 'max', 'mean', 'median', 'std']
features = ['Salary', 'Age', 'Experience_Years']
df1 = pd.DataFrame(columns=X)

for var in features:
    df1['min'] = df.groupby('Gender')[var].min()
    df1['max'] = df.groupby('Gender')[var].max()
    df1['mean'] = df.groupby('Gender')[var].mean()
    df1['median'] = df.groupby('Gender')[var].median()
    df1['std'] = df.groupby('Gender')[var].std()

X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, df1.loc['Female'], 0.4, label='Female')
plt.bar(X_axis + 0.2, df1.loc['Male'], 0.4, label='Male')
plt.xticks(X_axis, X)
plt.xlabel('Statistical information')
plt.ylabel('Value')
plt.title('Groupwise Statistical Information of Employee Salary Dataset')
plt.legend()
plt.show()
