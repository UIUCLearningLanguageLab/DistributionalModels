import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(df, groupby_columns, y_column, yerr_column, title, y_label, line_properties=None):
    # Font size constants
    TITLE_FONTSIZE = 16
    AXIS_LABEL_FONTSIZE = 12
    TICK_LABEL_FONTSIZE = 10
    LEGEND_FONTSIZE = 10

    sns.set_style("white")

    # Check if the groupby_columns list is valid
    if not groupby_columns or not all(col in df.columns for col in groupby_columns):
        raise ValueError("Invalid groupby_columns list")

    # Check if y_column and yerr_column are in the DataFrame
    if y_column not in df.columns or yerr_column not in df.columns:
        raise ValueError("y_column or yerr_column not found in DataFrame")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group the DataFrame
    grouped_df = df.groupby(groupby_columns)

    # Iterate over the line_properties dictionary
    for group_label_key, (custom_label, color, linewidth, linestyle) in line_properties.items():
        # Convert the string key back to a tuple if necessary
        group_values = tuple(group_label_key.split(', ')) if ', ' in group_label_key else group_label_key

        # Check if the group exists in the DataFrame
        if group_values in grouped_df.groups:
            group = grouped_df.get_group(group_values)

            # Plot the line
            ax.plot(group['epoch'], group[y_column], label=custom_label, color=color, linewidth=linewidth, linestyle=linestyle)

            # Add shaded error (confidence interval)
            ax.fill_between(group['epoch'], group[y_column] - group[yerr_column], group[y_column] + group[yerr_column], color=color, alpha=0.3)

    # Adding labels and title with specified font sizes
    ax.set_xlabel('Epoch', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # Adjusting tick labels font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    # Remove grid lines
    ax.grid(False)

    # Creating a legend with a specific font size
    ax.legend(title=' & '.join(groupby_columns), fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
