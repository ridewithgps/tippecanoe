import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats

def main():
    # Set up the argument parser to allow for a file input
    parser = argparse.ArgumentParser(description="Analyze zoom level occurrences in CSV data.")
    parser.add_argument('--file', type=str, default="data.csv", help="Path to the CSV file")
    args = parser.parse_args()

    # Read the CSV file (expects header "idx, zoom")
    df = pd.read_csv(args.file)
    
    # Reset the DataFrame index so we can use row numbers to compute distances
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'row'}, inplace=True)
    
    # Group the data by zoom level and compute statistics for each group
    results = {}
    groups = df.groupby('zoom')
    
    for zoom, group in groups:
        count = len(group)
        # Use the row numbers (which reflect the order in the CSV) to compute distances
        rows = np.sort(group['row'].values)
        if count > 1:
            diffs = np.diff(rows)
            avg_diff = np.mean(diffs)
            std_diff = np.std(diffs)  # population standard deviation
            diff_list = diffs.tolist()
        else:
            avg_diff = None
            std_diff = None
            diff_list = []
        results[zoom] = {
            'count': count,
            'avg_diff': avg_diff,
            'std_diff': std_diff,
            'diffs': diff_list
        }
    
    # Calculate overall statistics for the counts across zoom levels
    counts = np.array([results[z]['count'] for z in results])
    overall_mean_count = np.mean(counts)
    overall_std_count = np.std(counts)
    
    # Print text output sorted by zoom level (low to high)
    print("Zoom Level Statistics:")
    print(f"Overall mean count: {overall_mean_count:.2f}")
    print(f"Overall std of counts: {overall_std_count:.2f}\n")
    
    for zoom in sorted(results.keys()):
        res = results[zoom]
        print(f"Zoom level {zoom}: Count = {res['count']}", end='')
        if res['avg_diff'] is not None:
            print(f", Average row distance = {res['avg_diff']:.2f}, Std = {res['std_diff']:.2f}")
        else:
            print(" (only one occurrence, no distance data)")
    
    # --- Plotting ---
    # Figure 1: Two subplots
    zoom_levels = sorted(results.keys())
    count_values = [results[z]['count'] for z in zoom_levels]
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Bar chart of counts per zoom level
    plt.subplot(1, 2, 1)
    plt.bar(zoom_levels, count_values, color='skyblue')
    plt.xlabel("Zoom Level")
    plt.ylabel("Count")
    plt.title("Count per Zoom Level")
    
    # Subplot 2: Boxplot of row distance distributions (only for zoom levels with >1 occurrence)
    zooms_with_diffs = [z for z in zoom_levels if len(results[z]['diffs']) > 0]
    data = [results[z]['diffs'] for z in zooms_with_diffs]
    
    plt.subplot(1, 2, 2)
    plt.boxplot(data, labels=zooms_with_diffs)
    plt.xlabel("Zoom Level")
    plt.ylabel("Row Distance")
    plt.title("Distance Distribution per Zoom Level")
    
    plt.tight_layout()
    plt.show()
    
    # # Figure 2: For each zoom level with multiple occurrences, plot a histogram with a fitted normal curve.
    # num_plots = len(zooms_with_diffs)
    # if num_plots > 0:
    #     ncols = 3  # Set number of columns in the subplot grid
    #     nrows = (num_plots + ncols - 1) // ncols  # Compute number of rows needed
    #     plt.figure(figsize=(ncols * 5, nrows * 4))
    #     for i, z in enumerate(zooms_with_diffs):
    #         diffs = np.array(results[z]['diffs'])
    #         mu = np.mean(diffs)
    #         sigma = np.std(diffs)
    #         plt.subplot(nrows, ncols, i + 1)
    #         # Plot histogram of row distances
    #         plt.hist(diffs, bins=10, density=True, alpha=0.6, color='g', edgecolor='black')
    #         # Create a range of x values and compute the normal PDF with the calculated mu and sigma
    #         x = np.linspace(min(diffs), max(diffs), 100)
    #         pdf = stats.norm.pdf(x, mu, sigma)
    #         plt.plot(x, pdf, 'k', linewidth=2)
    #         plt.title(f'Zoom {z}\n(mu={mu:.2f}, sigma={sigma:.2f})')
    #         plt.xlabel('Row Distance')
    #         plt.ylabel('Density')
    #     plt.tight_layout()
    #     plt.suptitle("Row Distance Distributions with Normal Fit", y=1.02)
    #     plt.show()

if __name__ == "__main__":
    main()
