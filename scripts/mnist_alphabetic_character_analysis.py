import numpy as np
from helper_functions import *

############# 
# LOAD DATA #
#############

# See /home/daniel/git/work-in-progress/scripts/handwritten_characters/English_Handwritten_Characters_Distances_To_Centroids.py
# for the code that generated the data_np and centroids files 

# load data_np and centroids
data_np = np.load('/home/daniel/git/work-in-progress/english_handwritten_characters_distances.npy')
# array([9.97543097e-01, 1.33918874e-07, 5.63467984e-05, 4.57849190e-08,
#       1.05705431e-05, 5.70987027e-07, 1.13680355e-04, 1.48702784e-05,
#       5.67646348e-05, 2.20390572e-03, 0.00000000e+00, 0.00000000e+00,
#       6.83514517e-03, 6.83514517e-03])
# Columns indexes 0-9, softmax output
# Column index 10, true class label 0-61, 0-9 are digits, 10-61 are letters A-Z, a-z
# Column index 11, predicted class label 0-9, always a digit as dataset was trained on MNIST
# Column index 12, distance to true class centroid
# Column index 13, distance to predicted class centroid

centroids = np.load('/home/daniel/git/work-in-progress/centroids.npy')

#######################################################
# BAR CHARTS, NEAREST MISCLASSIFICATIONS TO CENTROIDS #
# TRUE CLASS AND PREDICTED CLASS                      #
#######################################################

thresholds_same, indexes_same = find_thresholds_digits_only(data_np, same_class=True)

# For points incorrectly classified as class i
thresholds_different, indexes_different = find_thresholds_digits_only(data_np, same_class=False)

########################################
# PLOT THRESHOLD BARCHART SIDE BY SIDE #
########################################

plot_thresholds_comparison(thresholds_same, thresholds_different, prefix="English Handwritten Characters Digits Only ", filename="../figures/english_handwritten_characters_digits_only_thresholds.png", save=True)
# Saved as figures/english_handwritten_characters_digits_only_thresholds.png

####################
# CONFUSION MATRIX #
####################

# Confusion matrix for digits only
confusion_matrix_digits = confusion_matrix(data_np, digits_only=True)
plot_confusion_matrix(confusion_matrix_digits, prefix="English Handwritten Characters Digits Only ", filename="../figures/english_handwritten_characters_digits_only_confusion_matrix.png", save=True)
# Saved as figures/english_handwritten_characters_digits_only_confusion_matrix.png

################################################
# ALPHABETIC CHARACTERS DISTANCES TO CENTROIDS #
################################################

alpha_thresholds, alpha_indexes, alpha_avg_distances = find_thresholds_alphabetic_only(data_np)

plot_alphabetic_distances(alpha_thresholds, alpha_avg_distances, prefix="English Handwritten Alphabetic Characters ", filename="../figures/english_handwritten_characters_alphabetic_only_thresholds.png", save=True)
# Saved as figures/english_handwritten_characters_alphabetic_only_thresholds.png

######################################### 
# "MISCLASSIFIED" ALPHABETIC CHARACTERS #
#########################################

directory = '/home/daniel/.cache/kagglehub/datasets/dhruvildave/english-handwritten-characters-dataset/versions/3/Img'
png_files = get_png_files(directory)

# Plot the closest alphabetic characters to each digit
plot_closest_letters_to_digits(alpha_indexes, directory, png_files, alpha_thresholds, alpha_avg_distances, data_np, filename="../figures/closest_alphabetic_characters_to_each_digit.png", save=True)
# Saved distance comparison plot as figures/closest_alphabetic_characters_to_each_digit.png

# TODO
#1. Fix character plot_softmax
#2. Plot average softmax output for each digit
#2.1 Plot average softmax output for misclassified digits and alphabetic characters, one above the other.
#3. Table with thresholds and misclassification counts.
#4. Confusion matrix for alphabetic characters x digits TBC
#5. Need to combine MNIST figures with English handwritten characters, to figure out how accuracy is affected by the addition of alphabetic characters.

