# ---------------------------------------------------------------------------------------------------
# Scientific Programming Final Project
# By: Linnaeus Bundalian, Judith Osuna, David Cabezas, Martin Kusasira Morgan, Sofía González. 

# Functions for ELT and final dataset creation
# ---------------------------------------------------------------------------------------------------

# Import libraries and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer
import warnings


# Rename variables
def rename_variables(df, dict_names):
    """
    Rename variables in a given dataframe.

    Parameters:
        df (pd.DataFrame): Original dataframe.
        dict_names (dict): Dictionary mapping current column names to new names.

    Returns:
        renamed_df (pd.DataFrame): Dataframe with renamed columns.
    """
    renamed_df = df.rename(columns=dict_names)
    return renamed_df



# Detect null values
def count_nulls(df):
    """
    Calculate and display the number of null values per column.

    Parameters:
        df (pd.DataFrame): Original dataframe.

    Returns:
        --
    """
    null_counts = df.isnull().sum()
    print("\nNumber of null values in the dataset:\n", null_counts)



# Handle outliers
def handle_outliers(df):
    """
    Identify outliers using quartiles and interquartile range, and handle them
    using the IterativeImputer function.

    Parameters:
        df (pd.DataFrame): Original dataframe with outliers.

    Returns:
        new_df (pd.DataFrame): New dataframe without outliers.
    """
    
    # Get the remaining numerical columns (excluding the status column)
    numeric_columns = df.select_dtypes(include=['float64', 'int64'])
    
    # Exclude the status column from the numerical columns
    status_column = 'Status'
    numeric_columns = [col for col in numeric_columns if col != status_column]

    # Calculate the quartiles (Q1, Q3)
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Identify the outliers
    outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR)))

    # Create the new df
    new_df = df.copy()
    
    # Replace the outliers with NaN
    new_df[outliers] = np.nan

    # Apply the Iterative Imputer to the numerical columns with missing values
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42) 
    
    # Fit and transform the numeric columns with missing values
    new_df[numeric_columns] = iterative_imputer.fit_transform(new_df[numeric_columns])

    # Display how many outliers were replaced
    print("\nNumber of outliers handled per column:")
    for col in numeric_columns:
        print(f"{col}: {outliers[col].sum()}")

    return new_df



# Handle correlation between variables
def drop_correlated_vbles(df, threshold):
    """
    Identify correlated variable pairs and drop one of them.

    Parameters:
        df (pd.DataFrame): Original dataframe with correlated variables.
        threshold (float): The select threshold to filter correlation.

    Returns:
        reduced_df (pd.DataFrame): New dataframe without correlated variables.
        collinear_features (list): The correlated variables that were removed.
    """
    
    # Filter the numerical columns in the dataframe
    numeric_columns = df.select_dtypes(include=['float64'])

    # Calculate the correlation matrix only with the numerical columns
    corr_matrix = numeric_columns.corr().abs()

    # Get highly correlated variables
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1)  # Upper triangle mask
    upper_triangle_df = pd.DataFrame(upper_triangle, index=corr_matrix.index, columns=corr_matrix.columns)
    collinear_features = [
        column for column in corr_matrix.columns 
        if any(corr_matrix[column][upper_triangle_df[column] > 0] > threshold)
    ]

    # Drop the highly correlated variables
    reduced_df = df.drop(columns=collinear_features)
    
    return reduced_df, collinear_features



# Generate box plots. Used to visualize outliers and differences between controls and patients
def generate_boxplots(df, group_variable=None):
    """
    Generates boxplots for all numerical variables in the DataFrame.
    Box plots can be differentiated by a grouping variable or not.

    Parameters:
        df (pd.DataFrame): Pandas DataFrame containing the data.
        group_variable (str, optional): Name of the variable to group by.

    Returns:
        --
    """
    # Filter numerical columns from the DataFrame, excluding the group variable
    numerical_columns = df.select_dtypes(include=['number']).columns

    with warnings.catch_warnings(): # Suppress warnings within this block
        warnings.filterwarnings("ignore", category=FutureWarning) 
            
        # Create a plot for each numerical column
        for col in numerical_columns:
    
            if col != group_variable:
                plt.figure(figsize=(4, 3))
                
                # No grouping variable
                if group_variable is None:
                    sns.boxplot(y=df[col], color='skyblue')
                    plt.title(f"Boxplot of {col}")
        
                # Using a grouping variable
                else:
                    sns.boxplot(y=df[col], x=df[group_variable], color='skyblue')
                    plt.title(f"Boxplot of {col} by {group_variable}")
                
                plt.xlabel(col)
                plt.ylabel("Values")
                plt.show()



# Group by subject
def group_and_average(df):
    """
    Aggregate variables by averaging them for each subject after extracting the subject id.

    Parameters:
        df (pd.DataFrame): Original ungrouped dataframe.

    Returns:
        aggregated_df (pd.DataFrame): New dataframe grouped by subject.
    """
    # Create 'Subject_ID' column by extracting the subject part
    df['Subject_ID'] = df['Name'].str.extract(r'_(S\d+)_')[0]
    
    # Drop the 'Name' column
    df.drop(columns=['Name'], inplace=True)

    # Group by Subject_ID and calculate the mean
    aggregated_df = df.groupby('Subject_ID').mean().reset_index()

    # Check the aggregation
    print("Number of rows in the new df:", aggregated_df.shape[0])
    print("Number of unique subjects in the new df:", aggregated_df['Subject_ID'].nunique())

    return aggregated_df



# Scatter plot function
def scatter_plot(df, var1, var2, groups):
    """
    Create a scatter plot of two variables colored by a specified group.

    Parameters:
        df (pd.DataFrame): Dataframe.
        var1 (str): Variable for the x-axis.
        var2 (str): Variable for the y-axis.
        groups (str): Column defining groups.

    Returns:
        --
    """

    # Set figure size
    plt.figure(figsize=(4, 3))
    
    unique_groups = df[groups].unique()
    for group in unique_groups:
        subset = df[df[groups] == group]
        plt.scatter(subset[var1], subset[var2], label=group)

    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend()
    plt.title(f'Scatterplot of {var1} vs {var2} by {groups}')
    plt.show()



# Normalize data
def normalize_dataframe(df, method='z-score', exclude_columns=None):
    """
    Normalizes all variables in the DataFrame using either Z-score or Min-Max normalization.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        method (str): The normalization method, either 'z-score' or 'min-max'.
        exclude_columns (list): List of columns to exclude from normalization.

    Returns:
        df_normalized (pd.DataFrame): A DataFrame with normalized variables.
    """
    # Copy the dataframe
    df_normalized = df.copy()

    # Exclude specific columns from normalization
    if exclude_columns is None:
        exclude_columns = []
    
    with warnings.catch_warnings(): # Suppress warnings within this block
        warnings.filterwarnings("ignore", category=FutureWarning) 
            
        # Select columns to normalize (numerical columns except for excluded ones)
        columns_to_normalize = df_normalized.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_columns)
    
        # Apply normalization
        if method == 'z-score':
            scaler = StandardScaler()
            df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
        elif method == 'min-max':
            scaler = MinMaxScaler()
            df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
        else:
            raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max'.")

    return df_normalized

