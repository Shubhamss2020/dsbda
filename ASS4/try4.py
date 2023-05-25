import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def RemoveOutlier(df, var):
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

def DisplayOutlier(df, msg):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(msg)
    sns.boxplot(data=df, x='rm', ax=axes[0])
    sns.boxplot(data=df, x='lstat', ax=axes[1])
    fig.tight_layout()
    plt.show()

def display_dataset_info(df):
    # Display information about dataset
    print('Information of dataset:\n', df.info)
    print('Shape of dataset (row x column):', df.shape)
    print('Column names: ', df.columns)
    print('Total elements in dataset:', df.size)
    print('Datatypes of attributes:\n', df.dtypes)
    print('First 5 rows:\n', df.head().T)
    print('Last 5 rows:\n', df.tail().T)
    print('Any 5 rows:\n', df.sample(5).T)

def find_missing_values(df):
    # Find missing values
    print('Missing values:\n', df.isnull().sum())

def display_correlation_matrix(df):
    # Find correlation matrix
    print('Finding correlation matrix using heatmap: ')
    sns.heatmap(df.corr(), annot=True)
    plt.show()

def find_and_remove_outliers(df):
    # Finding and removing outliers
    print('Finding and removing outliers:')
    DisplayOutlier(df, 'Before removing outliers')
    print('Identifying outliers')
    df = RemoveOutlier(df, 'lstat')
    df = RemoveOutlier(df, 'rm')
    DisplayOutlier(df, 'After removing outliers')
    return df

def train_and_test_model(df):
    # Split the data into inputs and outputs
    x = df[['rm', 'lstat']]  # input data
    y = df['medv']  # output data

    # Training and testing data
    from sklearn.model_selection import train_test_split

    # Assign test data size 20%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Apply linear regression model on training data
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Display accuracy of the model
    from sklearn.metrics import mean_absolute_error

    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('Model score: ', model.score(x_test, y_test))

    # Test the model using user input
    print('Predict house price by giving user input: ')
    features = np.array([[6, 19]])
    prediction = model.predict(features)
    print('Prediction: {}'.format(prediction))

def menu():
    # Read dataset
    df = pd.read_csv('Boston.csv')
    print('Boston dataset loaded.')

    while True:
        print("\n-------- MENU --------")
        print("1. Display dataset information")
        print("2. Find missing values")
        print("3. Display correlation matrix")
        print("4. Find and remove outliers")
        print("5. Train and test the model")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")
        if choice == '1':
            display_dataset_info(df)
        elif choice == '2':
            find_missing_values(df)
        elif choice == '3':
            display_correlation_matrix(df)
        elif choice == '4':
            df = find_and_remove_outliers(df)
        elif choice == '5':
            train_and_test_model(df)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

menu()