import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def handle_outliers(df):
    DisplayOutliers(df, 'Before removing outliers')
    df = RemoveOutlier(df, 'raisedhands')
    df = RemoveOutlier(df, 'VisITedResources')
    df = RemoveOutlier(df, 'AnnouncementsView')
    df = RemoveOutlier(df, 'Discussion')
    DisplayOutliers(df, 'After removing outliers')
    return df

def label_encoding(df):
    df['gender'] = df['gender'].astype('category')
    df['gender'] = df['gender'].cat.codes
    print('Data types of gender after label encoding:', df.dtypes['gender'])
    print('Gender Values:', df['gender'].unique())
    return df

def plot_boxplot(df):
    sns.boxplot(data=df, x='gender', y='raisedhands', hue='gender')
    plt.title('Boxplot with 2 variables gender and raisedhands')
    plt.show()

def plot_multiple_boxplot(df):
    sns.boxplot(data=df, x='NationalITy', y='Discussion', hue='gender')
    plt.title('Boxplot with 3 variables Gender, Nationality and Discussion')
    plt.show()

def plot_scatterplot(df):
    print('Relationship between variables using Scatterplot:')
    sns.scatterplot(data=df, x='raisedhands', y='VisITedResources')
    plt.title('Scatterplot for raisedhands and VisITedResources')
    plt.show()

def apply_log_transformation(df):
    
    df['raisedhands_log'] = np.log(df['raisedhands'])
    # Visualize the original and transformed distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(data=df, x='raisedhands', kde=True, ax=axes[0])
    sns.histplot(data=df, x='raisedhands_log', kde=True, ax=axes[1])
    axes[0].set(title='Original Distribution')
    axes[1].set(title='Logarithmic Transformation')
    plt.show()

def menu():
    # Read dataset
    df = pd.read_csv('dataset.csv')
    print('Boston dataset loaded.')

    while True:
        print("\nMENU:")
        print("1. Display information about dataset")
        print("2. Display statistical information of numerical columns")
        print("3. Display null values")
        print("4. Handle outliers")
        print("5. Perform label encoding")
        print("6. Plot boxplot with gender and raisedhands")
        print("7. Plot boxplot with gender, nationality, and discussion")
        print("8. Plot scatterplot for raisedhands and VisITedResources")
        print("9. Apply logarithmic transformation to raisedhands")
        print("10. Exit")

        choice = input("Enter your choice (1-10): ")

        if choice == '1':
            print('Information of dataset:\n', df.info())
            print('Shape of dataset (row x column):', df.shape)
            print('Column names:', df.columns)
            print('Total elements in dataset:', df.size)
            print('Datatypes of attributes:\n', df.dtypes)
            print('First 5 rows:\n', df.head().T)
            print('Last 5 rows:\n', df.tail().T)
            print('Any 5 rows:\n', df.sample(5).T)
        elif choice == '2':
            print("Statistical information of Numerical Columns:\n", df.describe())
        elif choice == '3':
            print("Total Number of Null Values in Dataset:", df.isna().sum())
        elif choice == '4':
            df = handle_outliers(df)
        elif choice == '5':
            df = label_encoding(df)
        elif choice == '6':
            plot_boxplot(df)
        elif choice == '7':
            plot_multiple_boxplot(df)
        elif choice == '8':
            plot_scatterplot(df)
        elif choice == '9':
            apply_log_transformation(df)
        elif choice == '10':
            print("Exiting the program...")
            break
        else:
            print("Invalid choice! Please enter a valid option.")

menu()
