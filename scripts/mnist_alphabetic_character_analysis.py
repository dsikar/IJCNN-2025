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

plot_thresholds_comparison(thresholds_same, thresholds_different, prefix="English Handwritten Characters Digits Only ", filename="figures/english_handwritten_characters_digits_only_thresholds.png", save=True)
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
#4. Confusion matrix for alphabetic characters x digits TBC
#5. Need to combine MNIST figures with English handwritten characters, to figure out how accuracy is affected by the addition of alphabetic characters.

####################################################
# ALPHABETIC CHARACTERS PREDICTION THRESHOLDS DATA #
####################################################

# Claude code

# Create table of alphabetic characters showing prediction thresholds
alpha_threshold_table = create_alphabetic_prediction_threshold_table(data_np)

# Display the results
print("\nAlphabetic Character Prediction Threshold Table:")
print("-" * 120)
header = ["Threshold"] + [f"Digit {i}" for i in range(10)] + ["TBT", "TAT"]
print("{:<10} ".format(header[0]), end="")
for i in range(1, len(header)):
    print("{:<10} ".format(header[i]), end="")
print()
print("-" * 120)

for row in alpha_threshold_table:
    print("{:<10.2f} ".format(row[0]), end="")
    for i in range(1, len(row)):
        print("{:<10.0f} ".format(row[i]), end="")
    print()

# Save table as numpy array
np.save('../data/alphabetic_prediction_thresholds.npy', alpha_threshold_table)

# Alphabetic Character Prediction Threshold Table:
# ------------------------------------------------------------------------------------------------------------------------
# Threshold  Digit 0    Digit 1    Digit 2    Digit 3    Digit 4    Digit 5    Digit 6    Digit 7    Digit 8    Digit 9    TBT        TAT        
# ------------------------------------------------------------------------------------------------------------------------
# 0.90       528        365        242        85         493        239        223        108        517        60         2860       0          
# 0.80       528        365        242        85         493        239        223        108        517        60         2860       0          
# 0.70       518        359        236        81         481        229        212        104        508        58         2786       74         
# 0.60       497        333        217        68         461        205        195        93         459        51         2579       281        
# 0.50       470        304        201        52         440        184        174        87         414        47         2373       487        
# 0.40       442        272        181        44         420        157        153        75         365        37         2146       714        
# 0.30       407        238        167        38         397        141        124        61         317        33         1923       937        
# 0.20       365        206        147        25         368        122        98         48         253        27         1659       1201       
# 0.10       320        154        127        16         314        100        76         32         183        22         1344       1516       
# 0.05       272        125        111        10         274        77         57         22         140        18         1106       1754     

# Deep Seek solution

alpha_threshold_table = generate_table(data_np)

# Display the results
print("\nAlphabetic Character Prediction Threshold Table:")
print("-" * 120)
header = ["Threshold"] + [f"{i}" for i in range(10)] + ["TBT", "TAT"]
print("{:<10} ".format(header[0]), end="")
for i in range(1, len(header)):
    print("{:<10} ".format(header[i]), end="")
print()
print("-" * 120)

for row in alpha_threshold_table:
    print("{:<10.2f} ".format(row[0]), end="")
    for i in range(1, len(row)):
        print("{:<10.0f} ".format(row[i]), end="")
    print()

# Save table as numpy array
np.save('data/deepseek_alphabetic_prediction_thresholds.npy', alpha_threshold_table)

# ChatGPT solution

alpha_threshold_table = create_alphabetic_table(data_np)

# Display the results
print("\nAlphabetic Character Prediction Threshold Table:")
print("-" * 120)
header = ["Threshold"] + [f"{i}" for i in range(10)] + ["TBT", "TAT"]
print("{:<10} ".format(header[0]), end="")
for i in range(1, len(header)):
    print("{:<10} ".format(header[i]), end="")
print()
print("-" * 120)

for row in alpha_threshold_table:
    print("{:<10.2f} ".format(row[0]), end="")
    for i in range(1, len(row)):
        print("{:<10.0f} ".format(row[i]), end="")
    print()

# Save table as numpy array
np.save('data/chatgpt_alphabetic_prediction_thresholds.npy', alpha_threshold_table)

# Grok

alpha_threshold_table = create_alpha_threshold_table(data_np)

# Display the results
print("\nAlphabetic Character Prediction Threshold Table:")
print("-" * 120)
header = ["Threshold"] + [f"{i}" for i in range(10)] + ["TBT", "TAT"]
print("{:<10} ".format(header[0]), end="")
for i in range(1, len(header)):
    print("{:<10} ".format(header[i]), end="")
print()
print("-" * 120)

for row in alpha_threshold_table:
    print("{:<10.2f} ".format(row[0]), end="")
    for i in range(1, len(row)):
        print("{:<10.0f} ".format(row[i]), end="")
    print()

# Save table as numpy array
np.save('data/grok_0.8_0.02_alphabetic_prediction_thresholds.npy', alpha_threshold_table)


# Alphabetic Character Prediction Threshold Table:
# ------------------------------------------------------------------------------------------------------------------------
# Threshold  0          1          2          3          4          5          6          7          8          9          TBT        TAT        
# ------------------------------------------------------------------------------------------------------------------------
# 0.80       528        365        242        85         493        239        223        108        517        60         2860       0          
# 0.70       518        359        236        81         481        229        212        104        508        58         2786       74         
# 0.60       497        333        217        68         461        205        195        93         459        51         2579       281        
# 0.50       470        304        201        52         440        184        174        87         414        47         2373       487        
# 0.40       442        272        181        44         420        157        153        75         365        37         2146       714        
# 0.30       407        238        167        38         397        141        124        61         317        33         1923       937        
# 0.20       365        206        147        25         368        122        98         48         253        27         1659       1201       
# 0.10       320        154        127        16         314        100        76         32         183        22         1344       1516       
# 0.05       272        125        111        10         274        77         57         22         140        18         1106       1754       
# 0.02       228        101        71         8          240        56         30         18         12         10         774        2086       



# latex table generated from:
# https://chatgpt.com/c/67c44e41-82b4-800f-9352-9ff02e186a5b

###########################################################
# ALPHABETIC CHARACTERS PREDICTION THRESHOLDS LATEX TABLE #
###########################################################

# \begin{table}[ht]
# \centering
# \caption{Alphabetic Character Prediction Threshold Table: Percentages (top) and Numeric Values (bottom)}
# \label{tab:combined}
# \begin{tabular}{lccccccccccccc}
# \toprule
# \multicolumn{13}{c}{\textbf{Percentages (relative to baseline at 0.80)}} \\
# \midrule
# Thresh & 0 (\%) & 1 (\%) & 2 (\%) & 3 (\%) & 4 (\%) & 5 (\%) & 6 (\%) & 7 (\%) & 8 (\%) & 9 (\%) & TBT (\%) & TAT (\%) \\
# \midrule
# 0.80   & 100.0  & 100.0  & 100.0  & 100.0  & 100.0  & 100.0  & 100.0  & 100.0  & 100.0  & 100.0  & 100.0   & 0.0   \\
# 0.70   & 98.1   & 98.2   & 97.5   & 95.3   & 97.5   & 95.8   & 95.2   & 96.3   & 98.1   & 96.7   & 97.4    & 2.6   \\
# 0.60   & 94.2   & 91.2   & 89.7   & 80.0   & 93.5   & 85.8   & 87.5   & 86.1   & 88.8   & 85.0   & 90.2    & 9.8   \\
# 0.50   & 89.0   & 83.3   & 83.1   & 61.2   & 89.2   & 77.0   & 78.0   & 80.6   & 80.0   & 78.3   & 82.9    & 17.1  \\
# 0.40   & 83.7   & 74.5   & 74.8   & 51.8   & 85.3   & 65.6   & 68.6   & 69.4   & 70.6   & 61.7   & 75.0    & 25.0  \\
# 0.30   & 77.1   & 65.2   & 69.0   & 44.7   & 80.6   & 58.9   & 55.6   & 56.5   & 61.3   & 55.0   & 67.2    & 32.8  \\
# 0.20   & 69.2   & 56.4   & 60.7   & 29.4   & 74.7   & 51.1   & 44.0   & 44.4   & 48.9   & 45.0   & 58.0    & 42.0  \\
# 0.10   & 60.6   & 42.2   & 52.5   & 18.8   & 63.8   & 41.8   & 34.1   & 29.6   & 35.4   & 36.7   & 47.1    & 52.9  \\
# 0.05   & 51.5   & 34.3   & 45.9   & 11.8   & 55.6   & 32.2   & 25.6   & 20.4   & 27.1   & 30.0   & 38.7    & 61.3  \\
# 0.02   & 43.2   & 27.7   & 29.4   & 9.4    & 48.7   & 23.4   & 13.5   & 16.7   & 2.3    & 16.7   & 27.1    & 72.9  \\
# \midrule
# \multicolumn{13}{c}{\textbf{Numeric Values}} \\
# \midrule
# Thresh & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & TBT & TAT \\
# \midrule
# 0.80   & 528 & 365 & 242 & 85  & 493 & 239 & 223 & 108 & 517 & 60  & 2860 & 0   \\
# 0.70   & 518 & 359 & 236 & 81  & 481 & 229 & 212 & 104 & 508 & 58  & 2786 & 74  \\
# 0.60   & 497 & 333 & 217 & 68  & 461 & 205 & 195 & 93  & 459 & 51  & 2579 & 281 \\
# 0.50   & 470 & 304 & 201 & 52  & 440 & 184 & 174 & 87  & 414 & 47  & 2373 & 487 \\
# 0.40   & 442 & 272 & 181 & 44  & 420 & 157 & 153 & 75  & 365 & 37  & 2146 & 714 \\
# 0.30   & 407 & 238 & 167 & 38  & 397 & 141 & 124 & 61  & 317 & 33  & 1923 & 937 \\
# 0.20   & 365 & 206 & 147 & 25  & 368 & 122 & 98  & 48  & 253 & 27  & 1659 & 1201\\
# 0.10   & 320 & 154 & 127 & 16  & 314 & 100 & 76  & 32  & 183 & 22  & 1344 & 1516\\
# 0.05   & 272 & 125 & 111 & 10  & 274 & 77  & 57  & 22  & 140 & 18  & 1106 & 1754\\
# 0.02   & 228 & 101 & 71  & 8   & 240 & 56  & 30  & 18  & 12  & 10  & 774  & 2086\\
# \bottomrule
# \end{tabular}
# \label{app:alphabetic_mnist_cnn_misclassifications}
# \end{table}

#############################################
# ALPHABETIC CHARACTERS PREDICTION HEATMAPS #
#############################################

create_and_save_heatmaps(data_np, filename="../figures/english_handwritten_characters_x_confusion_matrix.png", save=True)
# Saved alphabetic misclassifications figures/english_handwritten_characters_x_confusion_matrix.png

#############################################
# NUMERIC CHARACTER SOFTMAX OUTPUT AVERAGES #
#############################################

digits_correct_match_condition = (data_np[:, 10] <= 9) & (data_np[:, 10] == data_np[:, 11])
correct_digit_predictions = data_np[digits_correct_match_condition]
digits_incorrect_match_condition = (data_np[:, 10] <= 9) & (data_np[:, 10] != data_np[:, 11])   
incorrect_digit_predictions = data_np[digits_incorrect_match_condition]

plot_digit_averages(correct_digit_predictions, incorrect_digit_predictions, color1='skyblue', color2='lightcoral', data="English Handwritten Characters Digits Only", title= "Softmax Average Distributions for Correct and Incorrect Digit Predictions", filename="figures/english_handwritten_characters_digit_softmax_averages.png", save=True)
# Saved average softmax outputs for correct and incorrect digit predictiions. figures/english_handwritten_characters_digit_softmax_averages.png
    
################################################
# ALPHABETIC CHARACTER SOFTMAX OUTPUT AVERAGES #
################################################