import pandas as pd
import os

# Define the function to add the new column with the constant value to each file
def add_new_column(file_path):
    last_digit = os.path.splitext(file_path)[0][-1]
        # person seated in the car with movements were saved with last digit 1, 2 or 3
    if last_digit == '1' or last_digit == '2' or last_digit == '3' :
        label_value = "1"
        # person seated in the car without movements were saved with last digit 4, 5 or 6
    elif last_digit == '4'or last_digit == '5' or last_digit == '6' :
        label_value = "1"
        # Dummy readings were saved with last digit 7
    elif last_digit == '7':
        label_value = "0"
        # Empty seat readings were saved with last digit 8
    else:
        label_value = "0"
        
    df = pd.read_excel(file_path)
    # Add a new column with the constant value to the DataFrame at the first position
    df.insert(loc=0, column='LABEL', value=label_value)        
    df.to_excel(file_path, index=False)

# Define a function to combine all xlsx files in the directory into a single dataframe
def combine_files(dir_path):
    all_data = pd.DataFrame()
    for filename in os.listdir(dir_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(dir_path, filename)
            df = pd.read_excel(file_path)
            all_data = pd.concat([all_data, df])
    return all_data

# Add the new column to each xlsx file in the directory
dir_path = 'C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\input\\'
for filename in os.listdir(dir_path):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(dir_path, filename)
        add_new_column(file_path)

# Combine all xlsx files into a single dataframe
combined_df = combine_files(dir_path)

# Write the combined dataframe to a new xlsx file
output_path = 'C:\\Users\\QuickPass\\Documents\\ML\\Sorted\\output\\combined2.xlsx'
combined_df.to_excel(output_path, index=False)