import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
import pandas as pd
from tqdm import tqdm

from model import model

from predict import recognize_digit
from preprocess import crop_image,cropping

image_dir = './20611830/'

# Change the current working directory to the image directory
os.chdir(image_dir)
filenames = os.listdir()
os.chdir('../')
filenames = [image_dir+filename for filename in filenames if filename.endswith(('.jpg', '.png'))]

output_dir1 = './'

csv_file_path = os.path.join(output_dir1, 'predicted_digits.csv')
cols=['S.No','Student','Q1a','Q1b','Q2a','Q2b','Q3a','Q3b','Q4a','Q4b','Q5a','Q5b','Q6a','Q6b',
      'Q7a','Q7b','Q8a','Q8b','Q9a','Q9b','Q10a','Q10b']
filenames_csv=['01a','01b','02a','02b','03a','03b','04a','04b','05a','05b','06a','06b',
      '07a','07b','08a','08b','09a','09b','10a','10b']
#df = pd.DataFrame(columns=cols)
data_rows = []
for i in tqdm(filenames):
    image_path = i
    top_left = (500, 384)
    bottom_right = (595, 650)

    # Crop the image
    cropped_image_left = crop_image(image_path, top_left, bottom_right)
    top_left = (770, 381)
    bottom_right = (861, 650)

    # Crop the image
    cropped_image_right = crop_image(image_path, top_left, bottom_right)

    gray_left = cropped_image_left[:,:,2]
    gray_left = -gray_left
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain binary image
    _, binary_left = cv2.threshold(gray_left, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    gray_right = cropped_image_right[:,:,2]
    gray_right = -gray_right
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain binary image
    _, binary_right = cv2.threshold(gray_right, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Assuming 'binary' is your original binary image data
    height, width = binary_left.shape[:2]

    # Define the number of rows and columns for cropping
    num_rows = 5
    num_cols = 2

    # Calculate the height and width of each cropped region
    crop_height = height // num_rows
    crop_width = width // num_cols

    # Define the specific rows and columns to display
    display_rows = [0,1, 2, 3, 4]
    display_columns = [0,1]


    output_dir = './resized_images'
    os.makedirs(output_dir, exist_ok=True)

    filename_map_left = {(0, 0): '01a',(0, 1): '01b',(1, 0): '03a',(1, 1): '03b',(2,0): '05a',(2,1): '05b',(3,0): '07a',(3,1): '07b',(4,0): '09a',(4,1): '09b'}
    cropping(display_rows,display_columns,crop_height,crop_width,binary_left,filename_map_left,output_dir)



    # Assuming 'binary' is your original binary image data
    height, width = binary_right.shape[:2]

    # Define the number of rows and columns for cropping
    num_rows = 5
    num_cols = 2

    # Calculate the height and width of each cropped region
    crop_height = height // num_rows
    crop_width = width // num_cols

    # Define the specific rows and columns to display
    display_rows = [0,1, 2, 3, 4]
    display_columns = [0,1]


    filename_map_right = {(0, 0): '02a',(0, 1): '02b',(1, 0): '04a',(1, 1): '04b',(2,0): '06a',(2,1): '06b',(3,0): '08a',(3,1): '08b',(4,0): '10a',(4,1): '10b'}

    cropping(display_rows,display_columns,crop_height,crop_width,binary_right,filename_map_right,output_dir)




    predictions_dict = {'S.No': len(data_rows) + 1, 'Student': i.split('/')[-1]}

    # Loop through each column in filenames_csv to process the images
    for csv_filename in filenames_csv:
        image_path = os.path.join(output_dir, csv_filename + '.png')

        # Call recognize_digit function to predict the digit
        predicted_digit = recognize_digit(image_path)
        predictions_dict[csv_filename] = predicted_digit

    # Append the current predictions to the list of data rows
    data_rows.append(predictions_dict)

df = pd.DataFrame(data_rows)
df.to_csv(csv_file_path, index=False)


df = pd.read_csv(csv_file_path)

# Reorder the columns based on the desired order
df = df[['S.No', 'Student'] + filenames_csv]

def calculate_total(row):
    total = 0
    columns = ['01a', '01b', '02a', '02b', '03a', '03b', '04a', '04b', '05a', '05b',
               '06a', '06b', '07a', '07b', '08a', '08b', '09a', '09b', '10a', '10b']

    for i in range(0, len(columns), 2):
        a_val = row[columns[i]] if not pd.isna(row[columns[i]]) else 0
        b_val = row[columns[i + 1]] if not pd.isna(row[columns[i + 1]]) else 0
        total += max(a_val + b_val, 0)  # Use max to handle NA as 0

    return total
df['total'] = df.apply(calculate_total, axis=1)
df.to_csv(csv_file_path, index=False)