import numpy as np
import matplotlib.pyplot as plt

def plot_digit_averages(train_correct_predictions, train_incorrect_predictions, color1='skyblue', color2='lightcoral', data="Training Data", title= "Softmax Average Distributions for Correct and Incorrect Digit Predictions", filename="", save=True):
    # Get the unique labels (digits) from column 11
    labels = np.unique(train_correct_predictions[:, 10]).astype(int)

    # Create a figure and subplots for each digit (2 rows: correct and incorrect predictions)
    fig, axs = plt.subplots(2, len(labels), figsize=(20, 10))
    fig.suptitle(f'{data} - {title}', fontsize=16)

    # Plot correct predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current digit
        digit_predictions = train_correct_predictions[train_correct_predictions[:, 10] == label, :10]
        # Calculate the average value for each index
        averages = np.mean(digit_predictions, axis=0)
        # Plot the bar graph for the current digit
        axs[0, i].bar(np.arange(10), averages, color=color1)
        # Set the y-axis to logarithmic scale
        axs[0, i].set_yscale('log')
        # Set the y-axis limits to start from 10^-4
        axs[0, i].set_ylim(bottom=1e-4)
        # Set the title for the current subplot
        axs[0, i].set_title(f'Digit {label} (Correct)', fontsize=12)
        # Set the x-tick positions and labels
        axs[0, i].set_xticks(np.arange(10))
        axs[0, i].set_xticklabels(np.arange(10), fontsize=10)
        # Add x-axis grid lines
        axs[0, i].set_xticks(np.arange(10), minor=True)
        axs[0, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add y-axis grid lines
        axs[0, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot incorrect predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current digit
        digit_predictions = train_incorrect_predictions[train_incorrect_predictions[:, 11] == label, :10]
        # Calculate the average value for each index
        averages = np.mean(digit_predictions, axis=0)
        # Plot the bar graph for the current digit
        axs[1, i].bar(np.arange(10), averages, color=color2)
        # Set the y-axis to logarithmic scale
        axs[1, i].set_yscale('log')
        # Set the y-axis limits to start from 10^-4
        axs[1, i].set_ylim(bottom=1e-4)
        # Set the title for the current subplot
        axs[1, i].set_title(f'Digit {label} (Incorrect)', fontsize=12)
        # Set the x-tick positions and labels
        axs[1, i].set_xticks(np.arange(10))
        axs[1, i].set_xticklabels(np.arange(10), fontsize=10)
        # Add x-axis grid lines
        axs[1, i].set_xticks(np.arange(10), minor=True)
        axs[1, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add y-axis grid lines
        axs[1, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Digit Index', ha='center', fontsize=14)

    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=14)

    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing for the main title

    # Display the plot
    plt.show()  

    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved average softmax outputs for correct and incorrect digit predictiions. {filename}")      

def plot_alphabetic_character_averages(train_incorrect_predictions, color='lightcoral', data="Training Data", title="Softmax Average Distributions for Incorrect Alphabetic Character Predictions", filename="", save=True):
    # Get the unique labels (alphabetic characters) from column 11
    # labels = np.unique(train_incorrect_predictions[:, 10]).astype(int)
    # hardcode the labels for the alphabetic characters
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create a figure and subplots for each alphabetic character (1 row: incorrect predictions)
    fig, axs = plt.subplots(1, len(labels), figsize=(20, 5))
    fig.suptitle(f'{data} - {title}', fontsize=16)

    # Plot incorrect predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current alphabetic character
        char_predictions = train_incorrect_predictions[train_incorrect_predictions[:, 11] == label, :10]
        # Calculate the average value for each index
        averages = np.mean(char_predictions, axis=0)
        # Plot the bar graph for the current alphabetic character
        axs[i].bar(np.arange(10), averages, color=color)
        # Set the y-axis to logarithmic scale
        axs[i].set_yscale('log')
        # Set the y-axis limits to start from 10^-4
        axs[i].set_ylim(bottom=1e-4)
        # Set the title for the current subplot
        axs[i].set_title(f'Digit {label} (Incorrect)', fontsize=12)
        # Set the x-tick positions and labels
        axs[i].set_xticks(np.arange(10))
        axs[i].set_xticklabels(np.arange(10), fontsize=10)
        # Add x-axis grid lines
        axs[i].set_xticks(np.arange(10), minor=True)
        axs[i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add y-axis grid lines
        axs[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Digit Index', ha='center', fontsize=14)

    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=14)

    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.85)  # Adjust the top spacing for the main title

    # Display the plot
    plt.show()  

    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved average softmax outputs for incorrect alphabetic character predictions. {filename}")        