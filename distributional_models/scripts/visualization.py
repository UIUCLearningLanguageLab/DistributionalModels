import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from adjustText import adjust_text

markers = ['o', 's', '^', 'v', 'd', 'p', 'h', '8', '>']
categories = ['Present A', 'Omitted A', 'Legal As', 'Illegal As', 'Present B', 'Omitted B', 'Legal Bs', 'Illegal Bs',
              'y']
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def main():
    confusion_matrix = np.random.rand(5, 5)
    x = np.linspace(0, 20, 11, dtype=int)
    data = np.random.rand(6, 11)

    # plot_confusion_matrix(matrix=confusion_matrix, vmin=0, vmax=1, xlabel='x', ylabel='y',
    #                       xticks=['ABCD','ABCD','ABCD','ABCD','ABCD'],
    #                       yticks=['ABCD','ABCD','ABCD','ABCD','ABCD'],
    #                       xtick_format=[10, 45], ytick_format=[10, 45])

    plot_lines(matrix=data, x=x, ymin=0, ymax=1, xticks=x, xtick_format=[10, 0], xlabel='epochs', ylabel='probability',
               title='fake prediction plot')


def plot_confusion_matrix(df=None, matrix=None, vmin=None, vmax=None,
                          xticks=None, yticks=None, xlabel=None, ylabel=None,
                          annot=True, xtick_format: Optional[List]=None, ytick_format: Optional[List]=None,title=None,
                          path=None):
    plt.figure(figsize=(10, 7))
    if df is not None:
        ax = sns.heatmap(df, annot=True, vmin=vmin, vmax=vmax, fmt='.1f')
    else:
        if matrix is not None:
            ax = sns.heatmap(matrix, xticklabels=xticks, yticklabels=yticks, annot=True, vmin=vmin, vmax=vmax, fmt='.1f')
    if xtick_format is not None:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize= xtick_format[0], rotation=xtick_format[1])
    if ytick_format is not None:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=ytick_format[0], rotation=ytick_format[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.tick_top()
    plt.title(title)
    plt.show()
    if path:
        plt.savefig(path)
        plt.clf()


def plot_lines(df=None, matrix=None, x=None, xcolumn=None, ycolumns: Optional[List[str]]=None, ymin=None, ymax=None,
                             xticks=None, yticks=None, xlabel=None, ylabel=None,
                             xtick_format: Optional[List]=None, ytick_format: Optional[List]=None, title=None,
                             path=None):
    if df:
        for i, ycolumn in enumerate(ycolumns):
            sns.lineplot(data=df, x=xcolumn, y=ycolumn, color=color_cycle[i])
    else:
        if matrix is not None and x is not None:
            texts = []
            for i, row in enumerate(matrix):
                plt.plot(x, row, color=color_cycle[i], label=categories[i])
                # Add labels near the end of each line
                text = plt.text(x[-1], row[-1], categories[i], color=color_cycle[i])
                texts.append(text)
        if xticks is not None and xtick_format is not None:
            plt.xticks(xticks, [str(i) for i in xticks], fontsize=xtick_format[0], rotation=xtick_format[1])
        if yticks is not None and ytick_format is not None:
            plt.yticks(yticks, [str(i) for i in yticks], fontsize=ytick_format[0], rotation=ytick_format[1])
        adjust_text(texts, ax= plt.gca(), expand_text=(1.2, 1.2), only_move={'points': 'xy', 'text': 'x'}, ensure_inside_axes=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ymin, ymax)
    plt.title(title)

    plt.show()
    if path:
        plt.savefig(path)
        plt.clf()

    # fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 10))
    # for column in range(2):
    #     if column == 0:
    #         grouped_df = sum_df.groupby(sum_df.input)
    #         title = 'Sum Output Activations'
    #     else:
    #         grouped_df = mean_df.groupby(mean_df.input)
    #         title = 'Mean Output Activations'
    #     for row in range(2):
    #         texts = []
    #         j = 0
    #         df_input = None
    #         if row == 0:
    #             df_input = grouped_df.get_group('A')
    #             axs[row, column].set_title(title, fontsize=15)
    #         else:
    #             df_input = grouped_df.get_group('y')
    #         for i, color in zip(range(len(categories)), color_cycle):
    #             label = categories[i]
    #             x = df_input.checkpoint
    #             y = df_input[label]
    #             axs[row, column].plot(x, y, label=label, color=color)
    #             pos_x = x.iloc[round(len(x)-1 * 0.8)]
    #             pos_y = y.iloc[round(len(x)-1 * 0.8)]
    #             texts.append(axs[row, column].annotate(label, xy=(pos_x, pos_y), fontsize=10, color=color))
    #         axs[row, column].set_ylabel('Output Activation', fontsize=13)
    #         axs[row, column].set_xlabel('Epochs', fontsize=13)
    #         # axs[row, column].legend(loc = 'upper right', prop={'size':10})
    #         axs[row, column].set_ylim(-0.1, 1.1)
    #         axs[row, column].set_yticks(np.arange(0, 1.1, 0.1))
    #         adjust_text(texts, ax=axs[row, column], expand_text=(1.2, 1.2), only_move={'points': 'xy', 'text': 'x'})
    #
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.9)
    # plt.savefig(fig_savepath)
    # plt.clf()


def visualization_with_tsne(embeddings, vocab):
    # get embeddings
    embeddings_df = pd.DataFrame(embeddings)

    # t-SNE transform
    tsne = TSNE(n_components=2)
    embeddings_df_trans = tsne.fit_transform(embeddings_df)
    embeddings_df_trans = pd.DataFrame(embeddings_df_trans)

    # get token order
    embeddings_df_trans.index = vocab

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=embeddings_df_trans[0],
            y=embeddings_df_trans[1],
            mode="text",
            text=embeddings_df_trans.index,
            textposition="middle center",
            textfont=dict(color="blue"),
        )
    )
    fig.write_html("../word2vec_visualization.html")





def plot_prediction_accuracy(sum_df, mean_df, fig_savepath):
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 10))
    for column in range(2):
        if column == 0:
            grouped_df = sum_df.groupby(sum_df.input)
            title = 'Sum Output Activations'
        else:
            grouped_df = mean_df.groupby(mean_df.input)
            title = 'Mean Output Activations'
        for row in range(2):
            texts = []
            j = 0
            df_input = None
            if row == 0:
                df_input = grouped_df.get_group('A')
                axs[row, column].set_title(title, fontsize=15)
            else:
                df_input = grouped_df.get_group('y')
            for i, color in zip(range(len(categories)), color_cycle):
                label = categories[i]
                x = df_input.checkpoint
                y = df_input[label]
                axs[row, column].plot(x, y, label=label, color=color)
                pos_x = x.iloc[round(len(x)-1 * 0.8)]
                pos_y = y.iloc[round(len(x)-1 * 0.8)]
                texts.append(axs[row, column].annotate(label, xy=(pos_x, pos_y), fontsize=10, color=color))
            axs[row, column].set_ylabel('Output Activation', fontsize=13)
            axs[row, column].set_xlabel('Epochs', fontsize=13)
            # axs[row, column].legend(loc = 'upper right', prop={'size':10})
            axs[row, column].set_ylim(-0.1, 1.1)
            axs[row, column].set_yticks(np.arange(0, 1.1, 0.1))
            adjust_text(texts, ax=axs[row, column], expand_text=(1.2, 1.2), only_move={'points': 'xy', 'text': 'x'})

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.9)
    plt.savefig(fig_savepath)
    plt.clf()


def plot_similarity_matrices(list_of_matrices, measure, checkpoint_lists, words, groups, fig_savepath):
    num_words = len(words)
    num_checkpoints = len(list_of_matrices)

    # Determine the layout of the subplots
    num_rows = int(np.ceil(np.sqrt(num_words // 2)))
    num_cols = int(np.ceil((num_words // 2) / num_rows))

    for fig_num, group in enumerate(groups):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        axes = axes.flatten()
        group_label = ['A', 'B', 'other']
        for i, word in enumerate(group):
            word_index = words.index(word)
            ax = axes[i]
            ax.set_title(f"Similarity of word {word} with other words")
            # Find the three most similar and three most dissimilar words in the last checkpoint
            similarities_last_checkpoint = list_of_matrices[-1][
                word_index].copy()  # Make a copy to avoid modifying the original matrix

            sorted_indices = np.argsort(similarities_last_checkpoint)
            top_similar_indices = sorted_indices[-6:-1]
            bottom_similar_indices = sorted_indices[:3]

            for j in np.concatenate((top_similar_indices, bottom_similar_indices)):
                similarities = [matrix[word_index, j] for matrix in list_of_matrices]
                ax.plot(checkpoint_lists, similarities, label=words[j])
            ax.set_ylim([-1, 1.25])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Similarity")
            ax.legend()

        # Remove extra subplots if any
        for i in range(len(group), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(fig_savepath + '/{}_similarity_plot_using_{}.png'.format(group_label[fig_num], measure))
        plt.clf()


def plot_cohyponym_score(df_w, df_h, fig_savepath):
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 10))
    for column in range(2):
        if column == 0:
            grouped_df = df_w.groupby(df_w.input)
            title = 'Paradigmatic Task Using Weights'
        else:
            grouped_df = df_h.groupby(df_h.input)
            title = 'Paradigmatic Task Using Hidden Activation'
        for row in range(2):
            df_input = None
            texts = []
            if row == 0:
                df_input = grouped_df.get_group('A')
                axs[row, column].set_title(title, fontsize=15)

            else:
                df_input = grouped_df.get_group('B')
            for i, color in zip(range(len(categories)), color_cycle):
                label = categories[i]
                x = df_input.checkpoint
                y = df_input[label]
                axs[row, column].plot(x, y, label=label, color=color)
                pos_x = x.iloc[round(len(x)-1 * 0.8)]
                pos_y = y.iloc[round(len(x)-1 * 0.8)]
                texts.append(axs[row, column].annotate(label, xy=(pos_x, pos_y), fontsize=10, color=color))
            axs[row, column].set_ylabel('Percentage', fontsize=13)
            axs[row, column].set_xlabel('Epochs', fontsize=13)
            # axs[row, column].legend(loc = 'upper right', prop={'size':10})
            axs[row, column].set_ylim(-0.1, 1.1)
            axs[row, column].set_yticks(np.arange(0, 1.1, 0.1))
            adjust_text(texts, ax=axs[row, column], expand_text=(1.2, 1.2), only_move={'points': 'xy', 'text': 'xy'})
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.9)
    plt.savefig(fig_savepath)
    plt.clf()


def plot_sim_score(df, type, fig_savepath):
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(10, 10))
    grouped_df = df.groupby(df.input)
    title = f'Paradigmatic Task Using {type}'
    for row in range(2):
        texts = []
        if row == 0:
            df_input = grouped_df.get_group('A')
            axs[row].set_title(title, fontsize=15)

        else:
            df_input = grouped_df.get_group('B')
        for i, color in zip(range(len(categories)), color_cycle):
            label = categories[i]
            x = df_input.checkpoint
            y = df_input[label]
            axs[row].plot(x, y, label=label, color=color)
            pos_x = x.iloc[round(len(x)-1 * 0.8)]
            pos_y = y.iloc[round(len(x)-1 * 0.8)]
            texts.append(axs[row].annotate(label, xy=(pos_x, pos_y), fontsize=10, color=color))
        axs[row].set_ylabel('Percentage', fontsize=13)
        axs[row].set_xlabel('Epochs', fontsize=13)
        # axs[row, column].legend(loc = 'upper right', prop={'size':10})
        axs[row].set_ylim(-1.1, 1.1)
        axs[row].set_yticks(np.arange(-1, 1.1, 0.1))
        adjust_text(texts, ax=axs[row], expand_text=(1.2, 1.2), only_move={'points': 'xy', 'text': 'xy'})
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.9)
    plt.savefig(fig_savepath)
    plt.clf()


if __name__ == "__main__":
    main()