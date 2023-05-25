import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('Employee_Salary_Dataset.csv')
print('Employee dataset loaded.')

# Function to display information about the dataset
def display_dataset_info():
    print('Information of dataset:\n', df.info())
    print('Shape of dataset (row x column):', df.shape)
    print('Column names:', df.columns)
    print('Total elements in dataset:', df.size)
    print('Datatypes of attributes:\n', df.dtypes)
    print('First 5 rows:\n', df.head().T)
    print('Last 5 rows:\n', df.tail().T)
    print('Any 5 rows:\n', df.sample(5).T)

# Function to display statistical information of numerical columns
def display_numerical_stats():
    columns = ['Experience_Years', 'Age', 'Salary']
    for column in columns:
        print('\n')
        print("****************************************  "+column+"  **************************************************")
        print(df[column].describe())
        print('\n')

# Function to display groupwise statistical summary
def display_groupwise_stats():
    columns = ['Experience_Years', 'Age', 'Salary']
    for column in columns:
        print('\n')
        print("****************************************  "+column+"  **************************************************")
        print(df.groupby('Gender')[column].describe().T)
        print('\n')

# Function to plot groupwise statistical information
def plot_groupwise_stats():
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

# Display options to the user
while True:
    print("Choose an option:")
    print("1. Display information about dataset")
    print("2. Display statistical information of numerical columns")
    print("3. Display groupwise statistical summary")
    print("4. Plot groupwise statistical information")
    print("5. Exit")
  
    # Get user's choice
    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        display_dataset_info()
    elif choice == '2':
        display_numerical_stats()
    elif choice == '3':
        display_groupwise_stats()
    elif choice == '4':
        plot_groupwise_stats()
    elif choice == '5':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please enter a valid option.")