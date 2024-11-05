import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.colors as mcolors


def plot_time_series(df, groupby_columns, y_column, yerr_column, title, y_label, ylim, line_properties=None,
                     save_path=None, accuracy=False):
    # Font size constants
    TITLE_FONTSIZE = 35 #16
    AXIS_LABEL_FONTSIZE = 35 #12
    TICK_LABEL_FONTSIZE = 25 #10
    LEGEND_FONTSIZE = 20 #10

    sns.set_style("white")

    # Check if the groupby_columns list is valid
    if not groupby_columns or not all(col in df.columns for col in groupby_columns):
        raise ValueError("Invalid groupby_columns list")

    # Check if y_column and yerr_column are in the DataFrame
    if y_column not in df.columns or yerr_column not in df.columns:
        raise ValueError("y_column or yerr_column not found in DataFrame")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 10))

    # Group the DataFrame
    grouped_df = df.groupby(groupby_columns)

    # Iterate over the line_properties dictionary
    for group_label_key, (custom_label, color, linewidth, linestyle) in line_properties.items():
        # Convert the string key back to a tuple if necessary
        group_values = tuple(group_label_key.split(', ')) if ', ' in group_label_key else group_label_key

        # Check if the group exists in the DataFrame
        if group_values in grouped_df.groups:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group = grouped_df.get_group(group_values)

            # Plot the line
            ax.plot(group['epoch'], group[y_column], label=custom_label, color=color, linewidth=linewidth, linestyle=linestyle)

            # Add shaded error (confidence interval)
            ax.fill_between(group['epoch'], group[y_column] - group[yerr_column], group[y_column] + group[yerr_column], color=color, alpha=0.3)
    if accuracy:
        ax.plot(group['epoch'], group['accuracy'], label='Accuracy', color='grey', linewidth=1, linestyle='solid')
        ax.plot(group['epoch'], group['alt_accuracy'], label='Alternative Accuracy', color='grey', linewidth=1, linestyle='dashdot')
    # Adding labels and title with specified font sizes
    ax.set_xlabel('Epoch', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # Adjusting tick labels font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.set_yticks(np.arange(ylim[0], ylim[1]+0.1, 0.2))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11))

    # Remove grid lines
    ax.grid(False)

    # Creating a legend with a specific font size
    # ax.legend(title=' & '.join(groupby_columns), fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def plot_sub_dendogram(ax, matrix, labels, method='ward'):
    def color_func(lbl):
        # if 'A1' in lbl:
        #     return 'red'
        # elif 'A2' in lbl:
        #     return 'orange'
        # elif 'B1' in lbl:
        #     return 'blue'
        # elif 'B2' in lbl:
        #     return 'purple'
        # else:
        return 'black'

    matrix = np.round(matrix, 4)
    condensed_matrix = squareform(matrix)
    linkage_matrix = linkage(condensed_matrix, method=method)
    dendro = dendrogram(linkage_matrix, labels=labels, ax=ax)
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(color_func(lbl.get_text()))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
    clusters = {i: [label] for i, label in enumerate(labels)}
    all_clusters = []
    for i, (cluster1, cluster2, dist, num_elements) in enumerate(linkage_matrix):
        cluster1 = int(cluster1)
        cluster2 = int(cluster2)

        # Merge clusters
        new_cluster = clusters[cluster1] + clusters[cluster2]

        # Update clusters
        clusters[len(matrix) + i] = new_cluster

        # Remove the old clusters
        del clusters[cluster1]
        del clusters[cluster2]

        # Print the members of the new cluster
        # print(f"Merge {i + 1}:")
        # print(f"Clusters {cluster1} and {cluster2} merged to form cluster with members: {new_cluster}")
        all_clusters.append(new_cluster)

    # At the end, clusters should contain the final clustering result
    # print("\nFinal clustering result:")
    # print(clusters)
    sorted_all_clusters = sorted(all_clusters, key=len)
    return dendro['leaves'], all_clusters


def plot_sub_heatmap(ax, matrix, order, labels=None):
    sorted_matrix = matrix[order, :][:, order]
    colors = ["white", "darkblue"]  # White for -1, dark blue for +1
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)
    # if np.min(matrix) < 0 and np.max(matrix) > 0:
    #     # Data contains both negative and positive values
    #     norm = mcolors.TwoSlopeNorm(vmin=np.min(matrix), vcenter=0, vmax=np.max(matrix))
    # elif np.min(matrix) >= 0:
    #     # Data is strictly non-negative
    #     norm = mcolors.LogNorm(vmin=np.min(matrix) + 1e-9, vmax=np.max(matrix))
    # else:
    #     # Data is strictly non-positive (rare case)
    #     norm = mcolors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
    cax = ax.matshow(sorted_matrix, cmap=cmap, vmin=np.min(matrix), vmax=np.max(matrix))
    if labels is not None:
        # Set tick positions
        sorted_labels = np.array(labels)[order]
        ax.set_xticks(np.arange(len(sorted_labels)))
        ax.set_yticks(np.arange(len(sorted_labels)))

        # Set tick labels
        ax.set_xticklabels(sorted_labels)
        ax.set_yticklabels(sorted_labels)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, ha="right")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # # Title and labels
    # ax.set_title('Alphabetically Sorted Heatmap of Similarity Matrix')
    # ax.set_xlabel('Items')
    # ax.set_ylabel('Items')

def plot_sub_dendo_heatmap(ax1, ax2, matrix, labels):

    # Plot the dendrogram and get the leaf order
    order = plot_sub_dendogram(ax1, matrix, labels)

    # Plot the heatmap using the order from the dendrogram
    plot_sub_heatmap(ax2, matrix, order)

    ax1.set_aspect('auto')
    ax2.set_aspect('auto')


def plot_sub_time_series(ax, df, groupby_columns, y_column, yerr_column, title, y_label, ylim, line_properties=None,
                         accuracy=False):
    # Font size constants
    # TITLE_FONTSIZE = 35 #16
    # AXIS_LABEL_FONTSIZE = 35 #12
    # TICK_LABEL_FONTSIZE = 25 #10
    # LEGEND_FONTSIZE = 20 #10

    sns.set_style("white")

    # Check if the groupby_columns list is valid
    if not groupby_columns or not all(col in df.columns for col in groupby_columns):
        raise ValueError("Invalid groupby_columns list")

    # Check if y_column and yerr_column are in the DataFrame
    if y_column not in df.columns or yerr_column not in df.columns:
        raise ValueError("y_column or yerr_column not found in DataFrame")

    # Group the DataFrame
    grouped_df = df.groupby(groupby_columns)

    # Iterate over the line_properties dictionary
    for group_label_key, (custom_label, color, linewidth, linestyle) in line_properties.items():
        # Convert the string key back to a tuple if necessary
        # Here, we use tuple(group_label_key.split(', ')) if splitting on commas results in multiple items
        group_values = tuple(group_label_key.split(', ')) if ', ' in group_label_key else (group_label_key,)

        # Check if the group exists in the DataFrame
        if group_values[0] in grouped_df.groups:
            # Pass the tuple group_values to get_group
            group = grouped_df.get_group(group_values)

            # Plot the line
            ax.plot(group['epoch'], group[y_column], label=custom_label, color=color, linewidth=linewidth, linestyle=linestyle)

            # Add shaded error (confidence interval)
            ax.fill_between(group['epoch'], group[y_column] - group[yerr_column], group[y_column] + group[yerr_column], color=color, alpha=0.3)
    if accuracy:
        ax.plot(group['epoch'], group['accuracy'], label='Accuracy', color='grey', linewidth=1, linestyle='solid')
        ax.plot(group['epoch'], group['alt_accuracy'], label='B specific Accuracy', color='grey', linewidth=1,
                linestyle='dashdot')

    # Adding labels and title with specified font sizes
    # ax.set_xlabel('Epoch', )
    # ax.set_ylabel(y_label, )
    # ax.set_title(title, )

    # Adjusting tick labels font size
    ax.tick_params(axis='both', which='major', )
    ax.set_yticks(np.arange(ylim[0], ylim[1]+0.1, 0.5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11))

    # Remove grid lines
    ax.grid(False)

    # Creating a legend with a specific font size
    # ax.legend(title=' & '.join(groupby_columns), fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_heatmaps(data_list, title_list, y_labels, cmap='viridis', figsize=(6, 3), save_path=None):
    # Number of rows is determined by the number of data sets
    global_min = min(data.min() for data in data_list)
    global_max = max(data.max() for data in data_list)

    num_plots = len(data_list)

    # Create a grid of subplots with one column and as many rows as there are data sets
    fig, axs = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots))

    plt.subplots_adjust(hspace=0.35)

    # If there is only one plot, axs is not a list, so we put it in a list
    if num_plots == 1:
        axs = [axs]

    for i, data in enumerate(data_list):
        # Plot heatmap
        sns.heatmap(data, ax=axs[i], linewidth=.5, annot=True, fmt='.2f', cbar=False, cmap=cmap, vmin=global_min, vmax=global_max, square=True)
        axs[i].set_title(title_list[i])
        axs[i].set_yticklabels(y_labels)
        axs[i].xaxis.tick_top()
        axs[i].xaxis.set_label_position('top')
        # axs[i].set_xticklabels(x_tick_list[i])
    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def plot_heat_map(array, y_labels, column_groups, title, color_list, hline_pos=None, save_path=None):
    TITLE_FONTSIZE = 35 #16
    AXIS_LABEL_FONTSIZE = 35 #12
    TICK_LABEL_FONTSIZE = 15
    LEGEND_FONTSIZE = 20 #10
    cmap = LinearSegmentedColormap.from_list("", color_list)
    # norm = plt.Normalize(-2, 2)
    num_groups = len(column_groups)
    columns_per_group = array.shape[1]/num_groups
    group_positions = [(i * columns_per_group + columns_per_group / 2 - 0.5) for i in range(num_groups)]

    plt.figure(figsize=(30, 30))
    ax = sns.heatmap(array, cmap='cividis', annot=True, fmt='.2f', yticklabels=y_labels, cbar=False, square=True)
    if hline_pos == None:
        ax.hlines([4, 8, 12], *ax.get_xlim(), colors='white', linewidth=4)
    else:
        ax.hlines(hline_pos, *ax.get_xlim(), colors='white', linewidth=4)
    ax.vlines(np.arange(0, array.shape[1], columns_per_group), *ax.get_ylim(), colors='white', linewidth=4)
    # ax.set_position([0.3, 0.1, 0.6, 0.8])
    plt.yticks(rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel('Hidden Units', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Input Sequences', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    plt.xticks(ticks=group_positions, labels=column_groups, rotation=0)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def plot_bars(num_bars, df_list, col_name):
    # Set the x locations and width of the bars
    x = np.arange(num_bars)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots()

    # Plot the first two bars sharing the same y-axis
    bars1 = ax1.bar(x-width, df_list[0][col_name], width, label='input')
    bars2 = ax1.bar(x, df_list[1][col_name], width, label='output')

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    # bars3 = ax2.bar(x + width + width / 2, df_list[0]['accuracy'], width, label='accuracy', color='g')

    # Labeling
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis 1', color='b')
    ax2.set_ylabel('Y-axis 2', color='g')

    # # Add ticks and labels
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(df1['x'])
    # ax1.legend([bars1, bars2], ['df1', 'df2'], loc='upper left')
    # ax2.legend([bars3], ['df3'], loc='upper right')

    plt.show()

def main():
    pass


if __name__ == "__main__":
    main()
