# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv('Placement_Data_Full_Class.csv')
print("Placement dataset is successfully loaded into DataFrame")

# Display menu
print("Menu:")
print("1. Display information of dataset")
print("2. Display shape of dataset")
print("3. Display column names")
print("4. Display number of elements in dataset")
print("5. Display first 5 rows")
print("6. Display last 5 rows")
print("7. Display any 5 random rows")
print("8. Display statistical information of numerical columns")
print("9. Display null values")
print("10. Convert data types")
print("11. Perform label encoding")
print("12. Perform normalization")
print("13. Exit")

# Function to display information of dataset
def display_info():
    print("Information of Dataset:\n", df.info())

# Function to display shape of dataset
def display_shape():
    print("Shape of Dataset (row * column):\n", df.shape)

# Function to display column names
def display_column_names():
    print("Columns names:\n", df.columns)

# Function to display number of elements in dataset
def display_num_elements():
    print("Number of elements in dataset:", df.size)

# Function to display first 5 rows
def display_first_5_rows():
    print("First 5 rows:\n", df.head())

# Function to display last 5 rows
def display_last_5_rows():
    print("Last 5 rows:\n", df.tail())

# Function to display any 5 random rows
def display_random_rows():
    print("Any 5 rows:\n", df.sample(5))

# Function to display statistical information of numerical columns
def display_stats():
    print("Statistical information of Numerical Columns:\n", df.describe())

# Function to display null values
def display_null_values():
    print("Total Number of Null Values in Dataset:", df.isna().sum())

# Function to convert data types
def convert_data_types():
    df['sl_no'] = df['sl_no'].astype('int8')
    print("Check Datatype of sl_no:", df.dtypes['sl_no'])
    df['ssc_p'] = df['ssc_p'].astype('int8')
    print("Check Datatype of ssc_p:", df.dtypes['ssc_p'])

# Function to perform label encoding
def perform_label_encoding():
    df['gender'] = df['gender'].astype('category')
    print("Data type of gender:", df.dtypes['gender'])
    df['gender'] = df['gender'].cat.codes
    print("Data type of gender after label encoding:\n", df['gender'])
    print("gender Values:\n", df["gender"].unique())

# Function to perform normalization
def perform_normalization():
    print("Normalization using Min-Max Feature Scaling:")
    df['salary'] = (df['salary'] - df['salary'].min()) / (df['salary'].max() - df['salary'].min())
    print(df.head().T)

# Main program
while True:
    choice = input("Enter your choice (1-13): ")

    if choice == '1':
        display_info()
    elif choice == '2':
        display_shape()
    elif choice == '3':
        display_column_names()
    elif choice == '4':
        display_num_elements()
    elif choice == '5':
        display_first_5_rows()
    elif choice == '6':
        display_last_5_rows()
    elif choice == '7':
        display_random_rows()
    elif choice == '8':
        display_stats()
    elif choice == '9':
        display_null_values()
    elif choice == '10':
        convert_data_types()
    elif choice == '11':
        perform_label_encoding()
    elif choice == '12':
        perform_normalization()
    elif choice == '13':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please enter a valid option.")