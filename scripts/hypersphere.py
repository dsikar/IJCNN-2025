##################
# hypersphere.py #
##################

# Aim: to study the hypherphere defined by 10 points in 10 dimensions,
# corresponding to certainty thresholds, of a 10-class classifier,
# where each point corresponds to a class threshold e.g. MNIST 0,1,2...9.

import numpy as np
import os
from helper_functions import *

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# MNIST dataset

# Load training results
train_data = np.load(os.path.join(current_dir, '../data/train_np.npy'))

# Load test results
test_data = np.load(os.path.join(current_dir, '../data/test_np_with_distances.npy'))

# Load centroids
centroids = np.load(os.path.join(current_dir, '../data/centroids.npy'))

# 1. Find uncertainty thresholds for each class
# We can define threshold as:
# 1.1. The distance from the centroid of the nearest misclassified point of the class, e.g.
# for class 6, the nearest 6 misclassified as 8.
# 1.2. The distance from the centroid of the nearest misclassified point, e.g. for class 6
# any misclassified 0,1,2,3,4,5,7,8,9 misclassified as 6

# train_data columns:
# train_data.shape
# (60000, 12)

# - Columns 0 to 9: Softmax output of the trained neural network's class prediction
# - Column 10: True class
# - Column 11: Predicted class

train_data_with_distances = calculate_distances(train_data, centroids)

# For points that belong to class i but were misclassified
thresholds_same, indexes_same = find_thresholds(train_data_with_distances, same_class=True)

# For points incorrectly classified as class i
thresholds_different, indexes_different = find_thresholds(train_data_with_distances, same_class=False)

for i in range(10):
    debug_point(i, train_data_with_distances, thresholds_same, indexes_same)
    debug_point(i, train_data_with_distances, thresholds_different, indexes_different, "Different class")


# === Debugging Same class ===
# threshold[0]
# 0.7036534764474226
# index[0]
# 47690.0
# train_data_with_distances[47690]
# [4.79539245e-01 8.57034536e-07 4.82059747e-01 3.83279003e-05
#  1.67881957e-08 2.73815658e-06 9.85988372e-05 1.17470080e-03
#  3.64644974e-02 6.21215266e-04 0.00000000e+00 2.00000000e+00
#  7.03653476e-01 6.91112833e-01]
# entropy:
# 1.2101160164092855

# === Debugging Different class ===
# threshold[0]
# 0.0040732765941975885
# index[0]
# 494.0
# train_data_with_distances[494]
# [9.95668054e-01 3.87550863e-08 2.85520491e-06 6.95309421e-10
#  7.91840427e-10 1.59054820e-04 4.16915072e-03 3.06618773e-07
#  5.19655146e-07 3.82701302e-08 6.00000000e+00 0.00000000e+00
#  1.40144972e+00 4.07327659e-03]
# entropy:
# 0.04127655075781909
#    (...)

###################
# THRESHOLD TABLE #
###################

# Compute statistics
stats = compute_classification_statistics(train_data_with_distances, thresholds_same, thresholds_different)
# Generate table
latex_table = generate_threshold_table(stats)
print(latex_table)

########################################
# PLOT THRESHOLD BARCHART SIDE BY SIDE #
########################################

plot_thresholds_comparison(thresholds_same, thresholds_different, filename="../figures/mnist_thresholds_comparison.png", save=True)

######################################
# PLOT ENTROPY BARCHART SIDE BY SIDE #
######################################
    

##################################################################################
# PLOT SOFTMAX OUTPUT, DISTANCES, ENTROPY, TRUE LABEL, PREDICTED LABEL AND IMAGE #
################################################################################## 

# class 0: nearest class label 0, misclassified as another class
#plot_softmax(train_data_with_distances, 47690)
# class 0: nearest of any class, misclassified as 0
#plot_softmax(train_data_with_distances, 494)

same_class_idx = 47690
different_class_idx = 494
img_arr = get_mnist_img_array('train', same_class_idx)
# plot img_arr
plot_image(img_arr, "Class 0, misclassified as another class")
img_arr = get_mnist_img_array('train', different_class_idx)
plot_image(img_arr, "Another class misclassified as zero"), 

plot_softmax_comparison(train_data_with_distances, same_class_idx, different_class_idx)


######################################
# PLOT ENTROPY BARCHART SIDE BY SIDE #
######################################   


#####################
# HYPERSPHERE EDGES #
#####################

# https://chat.deepseek.com/a/chat/s/7589e006-38e9-4397-b084-003e105b7eea

min_dist, max_dist, mean_dist = compute_centroid_distances(centroids)

print(f"Minimum distance: {min_dist:.4f}")
print(f"Maximum distance: {max_dist:.4f}")
print(f"Mean distance: {mean_dist:.4f}")

# Minimum distance: 1.3684
# Maximum distance: 1.4021
# Mean distance: 1.3886

edges = get_selected_rows(train_data_with_distances, indexes_same)

# Compute distances between centroids and edges
distances = [np.linalg.norm(c - e) for c, e in zip(centroids, edges)]
print("Edge-to-centroid distances:", [f"{d:.4f}" for d in distances])

import numpy as np
from scipy.optimize import minimize

# Given data
# Load centroids
centroids = np.load(os.path.join(current_dir, '../data/centroids.npy'))
edges = get_selected_rows(train_data_with_distances, indexes_same)
d = np.array([0.7037, 0.7247, 0.6962, 0.7139, 0.7139, 0.7066, 0.7068, 0.6960, 0.6852, 0.7178])

# Run optimization
c_opt, r_opt = fit_hypersphere_with_constraints(edges, centroids, d)

print(f"Optimized hypersphere center: {c_opt}")
print(f"Optimized radius: {r_opt:.4f}")       

###########################
# MORE THRESHOLD ANALYSIS #
###########################

# NB we have 3 chats for this data
# https://claude.ai/chat/e478a52c-8272-4aaa-a79d-6aa610de09c5
# https://claude.ai/chat/b0f0d529-22bd-4099-8ed7-7f14e2854e94
thresholds = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
threshold_stats = analyze_thresholds(train_data_with_distances, thresholds)
print_threshold_analysis(threshold_stats, thresholds)

# Visualise thresholds
fig = create_visualization(threshold_stats)
plt.show()

# Calculate distances
distances = compute_pairwise_distances(centroids)

stats = {
    'min': np.min(distances),
    'max': np.max(distances),
    'mean': np.mean(distances),
    'std': np.std(distances)
}

print(f"Minimum distance: {stats['min']:.6f}")
print(f"Maximum distance: {stats['max']:.6f}")
print(f"Mean distance: {stats['mean']:.6f}")
print(f"Standard deviation: {stats['std']:.6f}")

# Minimum distance: 1.368433
# Maximum distance: 1.402052
# Mean distance: 1.388593
# Standard deviation: 0.008503

medoid, medoid_idx = compute_meta_centroid_medoid(centroids)
print(f"\nMedoid centroid (index {medoid_idx}):")
print(medoid)
print(f"Sum: {np.sum(medoid):.6f}")
print(f"Average distance to original centroids: {np.mean([np.sqrt(np.sum((c - medoid)**2)) for c in centroids]):.6f}")

## store results

# Run analysis
meta_arithmetic = compute_meta_centroid_arithmetic(centroids)
meta_weighted = compute_meta_centroid_weighted(centroids)
meta_geometric = compute_meta_centroid_geometric(centroids)

print("Properties of meta-centroids:")
print("\nArithmetic mean centroid:")
print(meta_arithmetic)
print(f"Sum: {np.sum(meta_arithmetic):.6f}")

print("\nWeighted mean centroid:")
print(meta_weighted)
print(f"Sum: {np.sum(meta_weighted):.6f}")

print("\nGeometric mean centroid:")
print(meta_geometric)
print(f"Sum: {np.sum(meta_geometric):.6f}")

# Compare distances
print("\nAverage distances to original centroids:")
print(f"Arithmetic: {np.mean([np.sqrt(np.sum((c - meta_arithmetic)**2)) for c in centroids]):.6f}")
print(f"Weighted: {np.mean([np.sqrt(np.sum((c - meta_weighted)**2)) for c in centroids]):.6f}")
print(f"Geometric: {np.mean([np.sqrt(np.sum((c - meta_geometric)**2)) for c in centroids]):.6f}")

meta_centroid_avg_dist = np.mean([np.sqrt(np.sum((c - meta_geometric)**2)) for c in centroids])


meta_centroids = {
    'arithmetic': {
        'values': meta_arithmetic,
        'avg_distance': compute_avg_distance(meta_arithmetic, centroids)
    },
    'weighted': {
        'values': meta_weighted,
        'avg_distance': compute_avg_distance(meta_weighted, centroids)
    },
    'geometric': {
        'values': meta_geometric,
        'avg_distance': compute_avg_distance(meta_geometric, centroids)
    }
}

# add distance from every prediction to geometric meta centroid

train_data_with_meta = add_meta_centroid_distances(
    train_data_with_distances, 
    meta_centroids['geometric']['values']
)

cluster_stats = populate_cluster_stats(train_data_with_meta, centroids, meta_centroid_avg_dist)

for key in cluster_stats:
    print(f"\nCluster {key}:")
    print(cluster_stats[key])

validations = validate_cluster_stats(cluster_stats, train_data_with_meta)
print_validation_results(validations)


###############################
# ENHANCED THRESHOLD ANALYSIS #
###############################

import numpy as np

# Extract relevant columns
distances = np.array([sample[13] for sample in train_data_with_meta])
true_classes = np.array([sample[10] for sample in train_data_with_meta])
predicted_classes = np.array([sample[11] for sample in train_data_with_meta])

# Total correct/incorrect in the dataset
correct_mask = (true_classes == predicted_classes)
C_total = np.sum(correct_mask)
I_total = len(true_classes) - C_total

# Thresholds (sorted descendingly)
thresholds = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
results = []

for threshold in thresholds:
    retain_mask = (distances <= threshold)
    retained_correct = np.sum(correct_mask & retain_mask)
    retained_incorrect = np.sum(~correct_mask & retain_mask)
    total_retained = retained_correct + retained_incorrect
    
    # Calculate ratio as "X:1" (rounded to nearest integer)
    if retained_incorrect == 0:
        ratio = "∞:1"
    else:
        ratio_value = round(retained_correct / retained_incorrect)
        ratio = f"{ratio_value}:1"
    
    results.append({
        "Threshold": threshold,
        "Total Within Retention": total_retained,
        "Retention %": (total_retained / len(true_classes)) * 100,
        "Accuracy %": (retained_correct / total_retained) * 100,
        "Correct": retained_correct,
        "Incorrect": retained_incorrect,
        "Cum Lost": C_total - retained_correct,
        "Cum Elim": I_total - retained_incorrect,
        "Ratio": ratio
    })

# Print formatted table
print("Enhanced Threshold Analysis")
print("=" * 110)
print(f"{'Threshold':<10} {'Total':<8} {'Retention%':<10} {'Accuracy%':<10} {'Correct':<8} {'Incorrect':<8} {'CumLost':<8} {'CumElim':<8} {'Ratio':<8}")
print("-" * 110)
for res in results:
    print(f"{res['Threshold']:<10.2f} {res['Total Within Retention']:<8} {res['Retention %']:<10.2f} {res['Accuracy %']:<10.2f} "
          f"{res['Correct']:<8} {res['Incorrect']:<8} {res['Cum Lost']:<8} {res['Cum Elim']:<8} {res['Ratio']:<8}")
          

########################################
# META CENTROID HYPERSPHERE MEMBERSHIP #
########################################

import numpy as np

def compute_meta_centroid_counts(train_data_with_meta, meta_centroid_avg_dist):
    """
    Computes points inside the geometric meta centroid's hypersphere 
    as class thresholds decrease from 0.8 to 0.05.
    
    Args:
        train_data_with_meta: List/array of samples with column 14 being 
                             distance to geometric meta centroid
        meta_centroid_avg_dist: Precomputed average distance (0.9337...)
    
    Returns:
        List of dictionaries with threshold details and counts
    """
    # Extract column 14 (distance to meta centroid) for all samples
    distances_to_meta = np.array([sample[14] for sample in train_data_with_meta])
    
    # Define class thresholds in descending order
    class_thresholds = [0.80, 0.70, 0.60, 0.50, 
                       0.40, 0.30, 0.20, 0.10, 0.05]
    
    results = []
    
    for class_thresh in class_thresholds:
        # Calculate meta centroid threshold
        meta_thresh = meta_centroid_avg_dist - class_thresh
        
        # Count points inside the hypersphere
        points_inside = np.sum(distances_to_meta <= meta_thresh)
        
        results.append({
            "Class Threshold": class_thresh,
            "Meta Threshold": meta_thresh,
            "Points Inside": points_inside
        })
    
    return results

# Example usage
meta_results = compute_meta_centroid_counts(train_data_with_meta, meta_centroid_avg_dist=0.9337416966305699)

# Print formatted table
print("Meta Centroid Analysis")
print("==============================================")
print(f"{'Class Threshold':<15} | {'Meta Threshold':<15} | {'Points Inside':<12}")
print("----------------------------------------------")
for res in results:
    print(f"{res['Class Threshold']:<15.2f} | {res['Meta Threshold']:<15.4f} | {res['Points Inside']:<12}")     

###################################
# EXPAND TABLE WITH META ANALYSIS #
###################################

original_results = results

def expand_table_with_meta_analysis(original_results, meta_results):
    """
    Expands the original table with meta centroid metrics and calculates percentages.
    """
    # Sort results by threshold (descending)
    original_sorted = sorted(original_results, key=lambda x: x['Threshold'], reverse=True)
    meta_sorted = sorted(meta_results, key=lambda x: x['Class Threshold'], reverse=True)
    
    # Extract total dataset size and class-specific totals from the first row
    first_row = original_sorted[0]
    total_samples = 60000  # As per your data
    C_total = first_row['Correct'] + first_row['Cum Lost']  # Total correct in dataset
    I_total = first_row['Incorrect'] + first_row['Cum Elim']  # Total incorrect in dataset
    
    expanded_results = []
    
    for orig, meta in zip(original_sorted, meta_sorted):
        cum_lost = orig['Cum Lost']
        cum_elim = orig['Cum Elim']
        points_inside = meta['Points Inside']
        
        # Compute "No class" and percentages
        no_class = max((cum_lost + cum_elim) - points_inside, 0)
        pct_cum_lost = (cum_lost / C_total) * 100
        pct_cum_elim = (cum_elim / I_total) * 100
        pct_points_in = (points_inside / total_samples) * 100
        pct_no_class = (no_class / total_samples) * 100
        
        # Format percentages
        expanded_entry = {
            **orig,
            'Meta Threshold': meta['Meta Threshold'],
            'Points Inside': points_inside,
            'No class': no_class,
            '% CumLost': f"{pct_cum_lost:.4f}%",
            '% CumElim': f"{pct_cum_elim:.2f}%",
            '% PointsIn': f"{pct_points_in:.2f}%",
            '% No Class': f"{pct_no_class:.4f}%"
        }
        expanded_results.append(expanded_entry)
    
    return expanded_results

# Example usage
expanded_table = expand_table_with_meta_analysis(original_results, meta_results)

# Print formatted table
headers = [
    "Threshold", "Total", "Retention%", "Accuracy%", "Correct", "Incorrect",
    "CumLost", "CumElim", "Ratio", "MetaThresh", "PointsIn", "No class",
    "% CumLost", "% CumElim", "% PointsIn", "% No Class"
]
print("Enhanced Threshold Analysis with Meta Metrics")
print("=" * 160)
print(
    f"{headers[0]:<8} {headers[1]:<6} {headers[2]:<10} {headers[3]:<10} "
    f"{headers[4]:<8} {headers[5]:<10} {headers[6]:<8} {headers[7]:<8} "
    f"{headers[8]:<8} {headers[9]:<10} {headers[10]:<10} {headers[11]:<10} "
    f"{headers[12]:<10} {headers[13]:<10} {headers[14]:<10} {headers[15]:<10}"
)
print("-" * 160)
for res in expanded_table:
    print(
        f"{res['Threshold']:<8.2f} {res['Total Within Retention']:<6} {res['Retention %']:<10.2f} "
        f"{res['Accuracy %']:<10.2f} {res['Correct']:<8} {res['Incorrect']:<10} "
        f"{res['Cum Lost']:<8} {res['Cum Elim']:<8} {res['Ratio']:<8} "
        f"{res['Meta Threshold']:<10.4f} {res['Points Inside']:<10} {res['No class']:<10} "
        f"{res['% CumLost']:<10} {res['% CumElim']:<10} {res['% PointsIn']:<10} {res['% No Class']:<10}"
    )

def generate_latex_table(expanded_table):
    latex = """\\begin{table}[ht]
\\centering
\\caption{Enhanced Threshold Analysis with Meta Metrics. Key columns:
\\textbf{Thresh}: Class threshold;
\\textbf{Ret\\%}: Retention percentage;
\\textbf{Acc\\%}: Accuracy percentage;
\\textbf{Corr}: Correct predictions retained;
\\textbf{Incor}: Incorrect predictions retained;
\\textbf{CLost}: Cumulative correct lost;
\\textbf{CElim}: Cumulative incorrect eliminated;
\\textbf{Ratio}: Correct:Incorrect ratio;
\\textbf{MetaT}: Meta threshold;
\\textbf{PtsIn}: Points inside meta hypersphere;
\\textbf{NoCls}: Points outside both class and meta hyperspheres;
\\textbf{\\%CLst}: Percentage of total correct lost;
\\textbf{\\%CEm}: Percentage of total incorrect eliminated;
\\textbf{\\%Pts}: Percentage of dataset inside meta hypersphere;
\\textbf{\\%NCl}: Percentage of dataset in no-class zone.}
\\label{tab:threshold_analysis}
\\begin{tabular}{r r r r r r r r r r r r r r r r}
\\toprule
Thresh & Ret\\% & Acc\\% & Corr & Incor & CLost & CElim & Ratio & MetaT & PtsIn & NoCls & \\%CLst & \\%CEm & \\%Pts & \\%NCl \\\\
\\midrule
"""

    for res in expanded_table:
        # Preprocess percentage columns to replace % with \%
        pct_cum_lost = res['% CumLost'].replace('%', '\\%')
        pct_cum_elim = res['% CumElim'].replace('%', '\\%')
        pct_points_in = res['% PointsIn'].replace('%', '\\%')
        pct_no_class = res['% No Class'].replace('%', '\\%')
        
        line = (
            f"{res['Threshold']:.2f} & "
            f"{res['Retention %']:.2f}\\% & "
            f"{res['Accuracy %']:.2f}\\% & "
            f"{res['Correct']} & "
            f"{res['Incorrect']} & "
            f"{res['Cum Lost']} & "
            f"{res['Cum Elim']} & "
            f"{res['Ratio']} & "
            f"{res['Meta Threshold']:.4f} & "
            f"{res['Points Inside']} & "
            f"{res['No class']} & "
            f"{pct_cum_lost} & "
            f"{pct_cum_elim} & "
            f"{pct_points_in} & "
            f"{pct_no_class} \\\\\n"
        )
        latex += line

    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex

# Generate and print LaTeX code
latex_code = generate_latex_table(expanded_table)
print(latex_code)

###############################################
# RETENTION, ACCURACY, AND RATIO VS THRESHOLD #
###############################################

import matplotlib.pyplot as plt

# Extract data
thresholds = [res["Threshold"] for res in expanded_table]
retention = [res["Retention %"] for res in expanded_table]
accuracy = [res["Accuracy %"] for res in expanded_table]
ratios = [float(res["Ratio"].split(":")[0]) for res in expanded_table]
ratio_labels = [res["Ratio"] for res in expanded_table]

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(16, 10))

# Plot retention and accuracy on primary axis
retention_line = ax1.plot(thresholds, retention, marker='o', linestyle='-', color='tab:blue', label="Retention %")
accuracy_line = ax1.plot(thresholds, accuracy, marker='s', linestyle='--', color='tab:green', label="Accuracy %")
ax1.set_xlabel("Threshold (Distance to Predicted Class Centroid)", fontsize=14)
ax1.set_ylabel("Percentage (%)", fontsize=14)
ax1.invert_xaxis()  # Start from 0.8 (left) to 0.05 (right)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', labelsize=12)

# Create secondary y-axis for ratios
ax2 = ax1.twinx()
ratio_line = ax2.plot(thresholds, ratios, marker='^', linestyle=':', color='tab:red', label="Correct:Incorrect Ratio")
ax2.set_ylabel("Ratio (Correct:Incorrect)", fontsize=14)
ax2.tick_params(axis='y', labelsize=12)

# Add ratio labels ABOVE ratio points (red)
for t, ratio, label in zip(thresholds, ratios, ratio_labels):
    ax2.annotate(
        label, 
        xy=(t, ratio), 
        xytext=(0, 15),  # 15 points above
        textcoords='offset points', 
        ha='center', 
        va='bottom',
        fontsize=18,
        weight='bold',
        color='tab:red',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", lw=1, alpha=0.9)        
    )

# Add accuracy labels BELOW accuracy points (green)
for t, acc in zip(thresholds, accuracy):
    ax1.annotate(
        f"{acc:.2f}%", 
        xy=(t, acc), 
        xytext=(0, -25),  # 25 points below
        textcoords='offset points', 
        ha='center', 
        va='top',
        fontsize=18,
        weight='bold',
        color='tab:green',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:green", lw=1, alpha=0.9)
    )

# Add titles and legends
plt.title("Retention, Accuracy, and Ratio vs. Threshold", fontsize=16, pad=25)
ax1.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

ax1.yaxis.grid(True, linestyle='--', alpha=0.5)  # Already enabled
ax2.yaxis.grid(False)  # Disable for secondary axis to reduce clutter

# Adjust x-ticks
plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds], rotation=45, fontsize=12)

plt.tight_layout()
plt.show()

# ==================

import matplotlib.pyplot as plt

# Extract data
thresholds = [res["Threshold"] for res in expanded_table]
retention = [res["Retention %"] for res in expanded_table]
accuracy = [res["Accuracy %"] for res in expanded_table]
ratios = [float(res["Ratio"].split(":")[0]) for res in expanded_table]
ratio_labels = [res["Ratio"] for res in expanded_table]

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(16, 10))

# Plot retention and accuracy on primary axis
retention_line = ax1.plot(thresholds, retention, marker='o', linestyle='-', color='tab:blue', label="Retention %")
accuracy_line = ax1.plot(thresholds, accuracy, marker='s', linestyle='--', color='tab:green', label="Accuracy %")
ax1.set_xlabel("Threshold (Distance to Predicted Class Centroid)", fontsize=14)
ax1.set_ylabel("Percentage (%)", fontsize=14)
ax1.invert_xaxis()  # Start from 0.8 (left) to 0.05 (right)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', labelsize=12)

# Create secondary y-axis for ratios
ax2 = ax1.twinx()
ratio_line = ax2.plot(thresholds, ratios, marker='^', linestyle=':', color='tab:red', label="Correct:Incorrect Ratio")
ax2.set_ylabel("Ratio (Correct:Incorrect)", fontsize=14)
ax2.tick_params(axis='y', labelsize=12)

# Add ratio labels ABOVE ratio points (red)
for t, ratio, label in zip(thresholds, ratios, ratio_labels):
    ax2.annotate(
        label, 
        xy=(t, ratio), 
        xytext=(0, 15),  # 15 points above
        textcoords='offset points', 
        ha='center', 
        va='bottom',
        fontsize=18,
        weight='bold',
        color='tab:red',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", lw=1, alpha=0.9)  
    )

# Add accuracy labels BELOW accuracy points (green)
for t, acc in zip(thresholds, accuracy):
    ax1.annotate(
        f"{acc:.2f}%", 
        xy=(t, acc), 
        xytext=(0, -25),  # 25 points below
        textcoords='offset points', 
        ha='center', 
        va='top',
        fontsize=18,
        weight='bold',
        color='tab:green',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:green", lw=1, alpha=0.9)
    )

# Combine all legend entries into one, placed on the left-center
lines = retention_line + accuracy_line + ratio_line
labels = [line.get_label() for line in lines]
ax1.legend(
    lines, labels,
    loc='center left',          # Anchor point: center-left of the legend
    bbox_to_anchor=(0.02, 0.5), # Position 2% from left edge, vertically centered
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor='gray'
)

ax1.yaxis.grid(True, linestyle='--', alpha=0.5)  # Already enabled
ax2.yaxis.grid(False)  # Disable for secondary axis to reduce clutter

# Adjust x-ticks
plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds], rotation=45, fontsize=12)
plt.title("Retention, Accuracy, and Ratio vs. Threshold", fontsize=16, pad=25)
plt.tight_layout()
plt.show()

# ========================

import matplotlib.pyplot as plt

# Extract data
thresholds = [res["Threshold"] for res in expanded_table]
retention = [res["Retention %"] for res in expanded_table]
accuracy = [res["Accuracy %"] for res in expanded_table]
ratios = [float(res["Ratio"].split(":")[0]) for res in expanded_table]
ratio_labels = [res["Ratio"] for res in expanded_table]

# Create figure and primary axis (larger figure size for readability)
fig, ax1 = plt.subplots(figsize=(20, 12))

# Plot retention and accuracy on primary axis
retention_line = ax1.plot(thresholds, retention, marker='o', linestyle='-', color='tab:blue', label="Retention %")
accuracy_line = ax1.plot(thresholds, accuracy, marker='s', linestyle='--', color='tab:green', label="Accuracy %")

# Primary axis labels (bold and doubled font size)
ax1.set_xlabel("Threshold (Distance to Predicted Class Centroid)", fontsize=28, weight='bold')
ax1.set_ylabel("Percentage (%)", fontsize=28, weight='bold')
ax1.invert_xaxis()  # Start from 0.8 (left) to 0.05 (right)
ax1.grid(True, linestyle='--', alpha=0.7)

# Tick labels (bold and doubled size)
ax1.tick_params(axis='both', labelsize=24, width=2)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')

# Create secondary y-axis for ratios
ax2 = ax1.twinx()
ratio_line = ax2.plot(thresholds, ratios, marker='^', linestyle=':', color='tab:red', label="Correct:Incorrect Ratio")
ax2.set_ylabel("Ratio (Correct:Incorrect)", fontsize=28, weight='bold')
ax2.tick_params(axis='y', labelsize=24)
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')

# Add ratio labels ABOVE ratio points (bold, doubled size)
for t, ratio, label in zip(thresholds, ratios, ratio_labels):
    ax2.annotate(
        label, 
        xy=(t, ratio), 
        xytext=(0, 15),  # 15 points above
        textcoords='offset points', 
        ha='center', 
        va='bottom',
        fontsize=36,  # Doubled from 18 → 36
        weight='bold',
        color='tab:red',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", lw=2, alpha=0.9)
    )

# Add accuracy labels BELOW accuracy points (bold, doubled size)
for t, acc in zip(thresholds, accuracy):
    ax1.annotate(
        f"{acc:.2f}%", 
        xy=(t, acc), 
        xytext=(0, -25),  # 25 points below
        textcoords='offset points', 
        ha='center', 
        va='top',
        fontsize=36,  # Doubled from 18 → 36
        weight='bold',
        color='tab:green',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:green", lw=2, alpha=0.9)
    )

# Combine all legend entries (bold and doubled size)
lines = retention_line + accuracy_line + ratio_line
labels = [line.get_label() for line in lines]
ax1.legend(
    lines, labels,
    loc='center left',
    bbox_to_anchor=(0.02, 0.5),
    fontsize=24,  # Doubled from 12 → 24
    prop={'weight': 'bold'},  # Bold legend text
    frameon=True,
    framealpha=0.9,
    edgecolor='black'
)

# Title (bold and doubled size)
plt.title("Retention, Accuracy, and Ratio vs. Threshold", fontsize=32, pad=30, weight='bold')

# X-ticks (bold and doubled size)
plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds], rotation=45, fontsize=24, weight='bold')

plt.tight_layout()
plt.show()

# ======================

import matplotlib.pyplot as plt

# Extract data
thresholds = [res["Threshold"] for res in expanded_table]
retention = [res["Retention %"] for res in expanded_table]
accuracy = [res["Accuracy %"] for res in expanded_table]
ratios = [float(res["Ratio"].split(":")[0]) for res in expanded_table]
ratio_labels = [res["Ratio"] for res in expanded_table]

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(16, 10))

# Plot retention and accuracy on primary axis
retention_line = ax1.plot(thresholds, retention, marker='o', linestyle='-', color='tab:blue', label="Retention %")
accuracy_line = ax1.plot(thresholds, accuracy, marker='s', linestyle='--', color='tab:green', label="Accuracy %")
ax1.set_xlabel("Threshold (Distance to Predicted Class Centroid)", fontsize=14)
ax1.set_ylabel("Percentage (%)", fontsize=14)
ax1.invert_xaxis()  # Start from 0.8 (left) to 0.05 (right)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', labelsize=12)

# Create secondary y-axis for ratios
ax2 = ax1.twinx()
ratio_line = ax2.plot(thresholds, ratios, marker='^', linestyle=':', color='tab:red', label="Correct:Incorrect Ratio")
ax2.set_ylabel("Ratio (Correct:Incorrect)", fontsize=14)
ax2.tick_params(axis='y', labelsize=12)

# Add ratio labels ABOVE ratio points (red)
for t, ratio, label in zip(thresholds, ratios, ratio_labels):
    ax2.annotate(
        label, 
        xy=(t, ratio), 
        xytext=(0, 15),  # 15 points above
        textcoords='offset points', 
        ha='center', 
        va='bottom',
        fontsize=18,
        weight='bold',
        color='tab:red',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", lw=1, alpha=0.9)  
    )

# Add accuracy labels BELOW accuracy points (green)
for t, acc in zip(thresholds, accuracy):
    ax1.annotate(
        f"{acc:.2f}%", 
        xy=(t, acc), 
        xytext=(0, -25),  # 25 points below
        textcoords='offset points', 
        ha='center', 
        va='top',
        fontsize=18,
        weight='bold',
        color='tab:green',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:green", lw=1, alpha=0.9)
    )

# Combine all legend entries into one, placed on the left-center
lines = retention_line + accuracy_line + ratio_line
labels = [line.get_label() for line in lines]
ax1.legend(
    lines, labels,
    loc='center left',          # Anchor point: center-left of the legend
    bbox_to_anchor=(0.02, 0.5), # Position 2% from left edge, vertically centered
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor='gray'
)

ax1.yaxis.grid(True, linestyle='--', alpha=0.5)  # Already enabled
ax2.yaxis.grid(False)  # Disable for secondary axis to reduce clutter

# Adjust x-ticks
plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds], rotation=45, fontsize=12)
plt.title("Retention, Accuracy, and Ratio vs. Threshold", fontsize=16, pad=25)
plt.tight_layout()
plt.show()