import pandas as pd
import matplotlib.pyplot as plt



def weight_and_metric_split(df):
    # Unique weights and metrics to plot
    weights = df['weight'].unique()
    metrics = ['loss', 'precision', 'recall', 'f1']

    # Create subplots: one row per weight, one column per metric
    fig, axs = plt.subplots(len(weights), len(metrics), figsize=(20, 10), sharex=True)

    # Iterate over weights and metrics to fill in subplots
    for i, weight in enumerate(weights):
        df_weight = df[df['weight'] == weight]
        for j, metric in enumerate(metrics):
            ax = axs[i, j] if len(weights) > 1 else axs[j]  # Adjust for single row of subplots
            ax.plot(df_weight['epoch'], df_weight[f'train_{metric}'], label='Train', marker='o', linestyle='-')
            ax.plot(df_weight['epoch'], df_weight[f'test_{metric}'], label='Test', marker='x', linestyle='--')
            ax.set_title(f'Weight {weight} - {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()

def other_plot(df, change):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration
    metrics = ['test_loss', 'test_precision', 'test_recall', 'test_f1']
    metric_titles = ['Loss', 'Precision', 'Recall', 'F1 Score']

    # Plot data
    for i, metric in enumerate(metrics):
        for changed in df[f'{change}'].unique():
            # Filter the DataFrame to include only rows where epoch % 10 == 0
            subset = df[(df[f'{change}'] == changed) & (df['epoch'] % 10 == 0)]
            if not subset.empty:
                axes[i].plot(subset['epoch'], subset[metric], label=f'{changed} {changed}', marker='o')

        axes[i].set_title(metric_titles[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.split('_')[1].capitalize())
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_metric(df, metric,change):
    fig, ax = plt.subplots(figsize=(6, 5))
    metric_title = metric.replace('_', ' ').capitalize()
    
    for changed in df[f'{change}'].unique():
        subset = df[(df[f'{change}'] == changed) & (df['epoch'] % 10 == 0)]
        if not subset.empty:
            ax.plot(subset['epoch'], subset[metric], label=f'{change} {changed}', marker='o')
    
    ax.set_title(metric_title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.split('_')[1].capitalize())
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_metric_with_shaded_std(df, metric, change):
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_title = metric.replace('_', ' ').capitalize()
    
    for changed in df[change].unique():
        # Filter by `change` and only consider every 10th epoch
        subset = df[(df[change] == changed) & (df['epoch'] % 10 == 0)]
        
        # Aggregate by epoch, calculating mean and standard deviation over repetitions
        aggregated = subset.groupby('epoch')[metric].agg(['mean', 'std']).reset_index()
        
        if not aggregated.empty:
            # Plot the mean as a line
            ax.plot(aggregated['epoch'], aggregated['mean'], label=f'{change} {changed}', marker='o', linestyle='-')
            
            # Shade the area between mean + std and mean - std
            ax.fill_between(aggregated['epoch'], 
                            aggregated['mean'] - aggregated['std'], 
                            aggregated['mean'] + aggregated['std'], 
                            alpha=0.2)  # Adjust alpha for opacity
    
    ax.set_title(f"{metric_title} over Epochs by {change.capitalize()}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.split('_')[1].capitalize())
    ax.legend(title=change.capitalize())
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_accuracy_over_epochs(repeat_number, accuracy_csv_path='results/MPC/accuracy.csv'):
    df = pd.read_csv(accuracy_csv_path)
    
    # Calculating the mean and std deviation for each epoch across all repeats
    grouped = df.groupby('epoch').agg({'f1-score': ['mean', 'std'], 'precision': ['mean', 'std'], 'recall': ['mean', 'std']})
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    epochs = grouped.index.values
    f1_mean = grouped['f1-score_mean'].values
    f1_std = grouped['f1-score_std'].values
    precision_mean = grouped['precision_mean'].values
    precision_std = grouped['precision_std'].values
    recall_mean = grouped['recall_mean'].values
    recall_std = grouped['recall_std'].values

    plt.figure(figsize=(12, 8))
    
    # Plotting F1 Score Mean and Std Dev
    plt.plot(epochs, f1_mean, label='F1 Score Mean', color='blue')
    plt.fill_between(epochs, f1_mean-f1_std, f1_mean+f1_std, color='blue', alpha=0.2)
    
    # Plotting Precision Mean and Std Dev
    plt.plot(epochs, precision_mean, label='Precision Mean', color='green')
    plt.fill_between(epochs, precision_mean-precision_std, precision_mean+precision_std, color='green', alpha=0.2)
    
    # Plotting Recall Mean and Std Dev
    plt.plot(epochs, recall_mean, label='Recall Mean', color='red')
    plt.fill_between(epochs, recall_mean-recall_std, recall_mean+recall_std, color='red', alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Average Accuracy Metrics Over Epochs Across All Repeats')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rounded_final_f1_distribution(accuracy_csv_path='results/MPC/accuracy.csv'):
    df = pd.read_csv(accuracy_csv_path)
    
    # Identifying the final epoch
    final_epoch = df['epoch'].max()
    
    # Extracting rows corresponding to the final epoch
    final_epoch_data = df[df['epoch'] == final_epoch]
    
    # Rounding the F1-scores to two decimal points
    final_epoch_data['f1-score_rounded'] = final_epoch_data['f1-score'].round(2)
    
    # Tallying the rounded accuracies
    tally = final_epoch_data['f1-score_rounded'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 8))
    tally.plot(kind='bar')
    
    plt.xlabel('Rounded F1-Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rounded Final F1-Scores')
    plt.grid(axis='y')
    plt.show()

def plot_final_f1_histogram(accuracy_csv_path='results/MPC/accuracy.csv'):
    df = pd.read_csv(accuracy_csv_path)
    
    # Identifying the final epoch
    final_epoch = df['epoch'].max()
    
    # Extracting rows corresponding to the final epoch
    final_epoch_data = df[df['epoch'] == final_epoch]
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_epoch_data['f1-score'], bins=10, edgecolor='k', alpha=0.7)
    
    plt.xlabel('F1-Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Final F1-Scores')
    plt.grid(axis='y')
    plt.show()
def plot_final_f1_kde(accuracy_csv_path='results/MPC/accuracy.csv'):
    df = pd.read_csv(accuracy_csv_path)
    
    # Identifying the final epoch
    final_epoch = df['epoch'].max()
    
    # Extracting rows corresponding to the final epoch
    final_epoch_data = df[df['epoch'] == final_epoch]
    
    plt.figure(figsize=(10, 6))
    final_epoch_data['f1-score'].plot.kde()
    final_epoch_data['precision'].plot.kde()
    final_epoch_data['f1-score'].plot.kde()
    
    plt.xlabel('F1-Score')
    plt.title('Kernel Density Estimate of Final F1-Scores')
    plt.grid(axis='both')
    plt.show()
import seaborn as sns
def plot_final_metrics_histogram_and_kde(accuracy_csv_path='results/MPC/accuracy.csv'):
    df = pd.read_csv(accuracy_csv_path)
    
    # Identifying the final epoch
    final_epoch = df['epoch'].max()
    
    # Extracting rows corresponding to the final epoch
    final_epoch_data = df[df['epoch'] == final_epoch]

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(8, 6))

    # Plot F1-Score
    sns.histplot(final_epoch_data['f1-score'], bins=10, kde=True, color='skyblue', edgecolor='k', alpha=0.7, ax=axes[0])
    axes[0].set_title('Histogram and KDE of Final F1-Scores')
    axes[0].set_xlabel('F1-Score')
    axes[0].set_ylabel('Frequency')

    # Plot Recall
    sns.histplot(final_epoch_data['recall'], bins=10, kde=True, color='red', edgecolor='k', alpha=0.7, ax=axes[1])
    axes[1].set_title('Histogram and KDE of Final Recall Scores')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Frequency')

    # Plot Precision
    sns.histplot(final_epoch_data['precision'], bins=10, kde=True, color='green', edgecolor='k', alpha=0.7, ax=axes[2])
    axes[2].set_title('Histogram and KDE of Final Precision Scores')
    axes[2].set_xlabel('Precision')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_single_metric_histogram_and_kde(metric, accuracy_csv_path='results/MPC/accuracy.csv'):
    # Validating the input metric
    valid_metrics = ['f1-score', 'recall', 'precision']
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric. Please choose one of {valid_metrics}")
    
    df = pd.read_csv(accuracy_csv_path)
    
    final_epoch_data = df.sort_values('epoch').groupby('target').tail(1)
    print(final_epoch_data.head())
    plt.figure(figsize=(8, 6))
    print(final_epoch_data[metric].describe())

    # Plotting the chosen metric
    sns.set_style('whitegrid')
    
    sns.histplot(final_epoch_data[metric], bins=10,edgecolor='k', alpha=0.7, color='#426294' if metric == 'f1-score' else '#31c1ce' if metric == 'recall' else '#0f2f4f')
    
    plt.title(f'Histogram and KDE of Final {metric.capitalize()}')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Frequency')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

def plot_accuracy_v_stopping_method(df = pd.read_csv('results/Timings/training/accuracy.csv')):
    df['Stopping Method'] = df['repeat'].map({1: 'With Early Stopping', 0: 'Without Early Stopping'})

    final_epochs_df = df.sort_values(by='epoch').groupby(['target', 'repeat']).tail(1)

    # Now, plotting the box plot using the final epochs' data
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Stopping Method', y='f1-score', data=final_epochs_df)

    plt.title('Final Accuracy (F1-Score) vs Stopping Method')
    plt.ylabel('Final F1-Score')
    plt.xlabel('Stopping Method')

    # Display the plot
    plt.show()

def plot_accuracy_v_N(df = pd.read_csv('results/Timings/efficiency_vs_N.csv')):
    final_accuracy = df.sort_values('epoch').groupby(['target', 'N', 'repeat']).tail(1)


    plt.figure(figsize=(12, 6))
    sns.boxplot(x='N', y='f1-score', data=final_accuracy)
    plt.title('Final Epoch Accuracy (F1-Score) vs. N')
    plt.xlabel('N (Number of Control Points)')
    plt.ylabel('Final F1-Score')
    plt.show()
def plot_avg_epoch_v_N(df = pd.read_csv('results/Timings/efficiency_vs_N.csv')):
    avg_epochs = df.groupby(['N', 'repeat'])['epoch'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='N', y='epoch', data=avg_epochs)
    plt.title('Average Number of Epochs vs. N')
    plt.xlabel('N (Number of Control Points)')
    plt.ylabel('Average Number of Epochs')
    plt.show()
def plot_n_vs_time_per_epoch(df = pd.read_csv('results/Timings/efficiency_vs_N.csv')):
    """
    Plots N vs average time per epoch based on the DataFrame provided.

    Args:
        df (pd.DataFrame): DataFrame containing the columns 'N', 'global_search_time', 'mpc_time', and 'epoch'.
    """
    # Calculate the total time for each run (global search time + MPC optimization time)
    df['total_time'] = df['global_search_time'] + df['mpc_time']
    
    # Calculate total epochs for each run to find the average time per epoch
    df['total_epochs'] = df.groupby(['target', 'N', 'repeat'])['epoch'].transform('max') + 1  # Adding 1 because epochs are zero-indexed
    
    # Calculate the average time per epoch for each run
    df['time_per_epoch'] = df['total_time'] / df['total_epochs']
    
    # Aggregate the average time per epoch by N
    avg_time_per_epoch_by_n = df.groupby('N')['time_per_epoch'].mean().reset_index()
    
    # Plotting
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='N', y='time_per_epoch', data=avg_time_per_epoch_by_n)
    
    plt.title('Average Time per Epoch vs. N')
    plt.xlabel('N (Number of Control Points)')
    plt.ylabel('Average Time per Epoch (seconds)')
    plt.grid(axis='y')
    
    plt.show()
def plot_n_vs_timeperseq(df = pd.read_csv('results/Timings/efficiency_vs_N.csv')):
    """
    Plots N vs average time per epoch based on the DataFrame provided.

    Args:
        df (pd.DataFrame): DataFrame containing the columns 'N', 'global_search_time', 'mpc_time', and 'epoch'.
    """
    # Calculate the total time for each run (global search time + MPC optimization time)
    df['total_time'] = df['global_search_time'] + df['mpc_time']
    df['total_time'] = df['total_time']/(df['N']*10)
    # Calculate total epochs for each run to find the average time per epoch
    df['total_epochs'] = df.groupby(['target', 'N', 'repeat'])['epoch'].transform('max') + 1  # Adding 1 because epochs are zero-indexed
    
    # Calculate the average time per epoch for each run
    df['time_per_epoch'] = df['total_time'] / df['total_epochs']
    
    # Aggregate the average time per epoch by N
    avg_time_per_epoch_by_n = df.groupby('N')['time_per_epoch'].mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    sns.barplot(x='N', y='time_per_epoch', data=avg_time_per_epoch_by_n)
    
    plt.title('Average Time per Sequence Length per Epoch vs. N')
    plt.xlabel('N (Number of Control Points)')
    plt.ylabel('Average Time per Sequence Length per Epoch  (seconds)')
    plt.grid(axis='y')
    
    plt.show()
from scipy.stats import pearsonr
def plot_accuracy_v_target_num(df = pd.read_csv('results/Timings/training/accuracy.csv')):
    final_epoch_data = df.sort_values('epoch').groupby('target').tail(1)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='target', y='f1-score', data=final_epoch_data, palette='viridis')
    correlation_coef, p_value = pearsonr(final_epoch_data['target'], final_epoch_data['f1-score'])
    plt.text(0.70, 0.98, f'Pearson correlation: {correlation_coef:.2f}\nP-value: {p_value:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.title('Final Accuracy (F1-Score) vs Target Depth')
    plt.xlabel('Target Depth')
    plt.ylabel('Final F1-Score')
    plt.show()

def plot_mpc_timing(df=pd.read_csv('results/Timings/test/accuracy.csv')):
    # Calculate the total time for each run
    df['total_time'] = df['mpc_time'] + df['global_search_time']
    
    avg_time = df.groupby('target')['total_time'].mean()
    final_epoch_data = df.sort_values('epoch').groupby('target').tail(1)
    print(final_epoch_data['epoch'].mean())

import numpy as np
def plot_transfer():
    # Placeholder metric names and titles for the graphs
    metrics = ['test_loss', 'test_precision', 'test_recall', 'test_f1']
    titles = ['Test Loss vs Epoch', 'Test Precision vs Epoch', 'Test Recall vs Epoch', 'Test F1 Score vs Epoch']
    data=pd.read_csv('results/transfer_learning/transfer_learning_scratch.csv')
    finetuning_data=pd.read_csv('results/transfer_learning/transfer_learning_finetuning_unfrozen_decoder.csv')
    finetuning_data_2 = pd.read_csv('results/transfer_learning/transfer_learning_finetuning_frozen_lstm.csv')
    finetuning_data_lr = pd.read_csv('results/transfer_learning/transfer_learning_lowered_lr.csv')
    finetuning_data_lr_2 = pd.read_csv('results/transfer_learning/transfer_learning_lowered_lr_0.0001.csv')
    data= data[data['epoch']%10==0]
    finetuning_data_lr_2 = finetuning_data_lr[finetuning_data_lr['epoch']%10==0]

    finetuning_data = finetuning_data[finetuning_data['epoch']%10==0]
    finetuning_data_2 = finetuning_data_2[finetuning_data_2['epoch']%10==0]
    finetuning_data_lr = finetuning_data_lr[finetuning_data_lr['epoch']%10==0]
    print(data[data['test_f1']==data['test_f1'].max()]['epoch'], 'standard')
    print(finetuning_data[finetuning_data['test_f1']==finetuning_data['test_f1'].max()]['epoch'], 'unfrozen decoder')
    print(finetuning_data_2[finetuning_data_2['test_f1']==finetuning_data_2['test_f1'].max()]['epoch'], 'frozen lstm')
    print(finetuning_data_lr[finetuning_data_lr['test_f1']==finetuning_data_lr['test_f1'].max()]['epoch'], 'lowered lr')
    
    for i, metric in enumerate(metrics):
        plt.plot(data['epoch'], data[metric], label='Standard Training', marker='o')
        plt.plot(finetuning_data['epoch'], finetuning_data[metric], label='Fine-Tuning (Un-frozen decoder - frozen others)', marker='x')
        plt.plot(finetuning_data_2['epoch'], finetuning_data_2[metric], label='Fine-Tuning (Frozen LSTM - unfrozen others)', marker='x')
        plt.plot(finetuning_data_lr['epoch'], finetuning_data_lr[metric], label='Fine-Tuning (LR- 0.00001)', marker='x')
        #plt.plot(finetuning_data_lr_2['epoch'], finetuning_data_lr_2[metric], label='Fine-Tuning (LR- 0.0001)', marker='o')
        plt.title(titles[i])
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()

        plt.tight_layout()
        plt.show()

def mpc_transfers():
    # Setting up the plot again
    baseline_df = pd.read_csv('results/transfer_learning/mpc_tests/accuracy_baseline.csv')
    frozen_lstm_df = pd.read_csv('results/transfer_learning/mpc_tests/accuracy_frozen_lstm.csv')
    lower_lr_df = pd.read_csv('results/transfer_learning/mpc_tests/accuracy_lower_lr.csv')
    unfrozen_decoder_df = pd.read_csv('results/transfer_learning/mpc_tests/accuracy_unfrozen_decoder.csv')
    zero_shot_df = pd.read_csv('results/transfer_learning/mpc_tests/accuracy_final.csv')


    last_epoch_baseline = baseline_df.groupby('target').last().reset_index()
    last_epoch_frozen_lstm = frozen_lstm_df.groupby('target').last().reset_index()
    last_epoch_lower_lr = lower_lr_df.groupby('target').last().reset_index()
    last_epoch_unfrozen_decoder = unfrozen_decoder_df.groupby('target').last().reset_index()
    last_epoch_zero_shot = zero_shot_df.groupby('target').last().reset_index()
    metrics = ['loss', 'precision', 'recall', 'f1-score']
    model_names = ['Baseline', 'Frozen LSTM', 'Lower LR', 'Unfrozen Decoder','Zero-shot']

    # Combining data for easier plotting
    combined_data = pd.concat([
        last_epoch_baseline.assign(Model='Baseline'),
        last_epoch_frozen_lstm.assign(Model='Frozen LSTM'),
        last_epoch_lower_lr.assign(Model='Lower LR'),
        last_epoch_unfrozen_decoder.assign(Model='Unfrozen Decoder'),
        last_epoch_zero_shot.assign(Model='Zero-shot')
    ])
    sns.set_style('whitegrid')
    sns.set(font_scale=1.5)
    # Plotting boxplots for each metric again
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='Model', y=metric, data=combined_data)
        plt.title(f'{metric.capitalize()} Distribution by Model')
        plt.xlabel('Model Type')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()
def plot_model_v_length(df=pd.read_csv('results/model_evaluation/accuracy_v_length.csv')):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Metrics by Length')
    data = df.copy()
    # Loss Plot
    axs[0, 0].plot(data["length"], data["test_loss"], label="Test Loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Length")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # Precision Plot

    axs[0, 1].plot(data["length"], data["test_precision"], label="Test Precision")
    axs[0, 1].set_title("Precision")
    axs[0, 1].set_xlabel("Length")
    axs[0, 1].set_ylabel("Precision")
    axs[0, 1].legend()

    # Recall Plot
    axs[1, 0].plot(data["length"], data["test_recall"], label="Test Recall")
    axs[1, 0].set_title("Recall")
    axs[1, 0].set_xlabel("Length")
    axs[1, 0].set_ylabel("Recall")
    axs[1, 0].legend()

    # F1 Score Plot
    axs[1, 1].plot(data["length"], data["test_f1"], label="Test F1 Score")
    axs[1, 1].set_title("F1 Score")
    axs[1, 1].set_xlabel("Length")
    axs[1, 1].set_ylabel("F1 Score")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_model_v_length_ind(df=pd.read_csv('results/transfer_learning/accuracy_final.csv')):
    # Create a figure and axis for each metric on individual plots
    metrics = ['test_loss', 'test_precision', 'test_recall', 'test_f1']
    metric_titles = ['Loss', 'Precision', 'Recall', 'F1 Score']

    for i, metric in enumerate(metrics):
        plt.plot(df['length'], df[metric], marker='o')
        plt.title(f'{metric_titles[i]} vs Length')
        plt.xlabel('Length')
        plt.ylabel(metric_titles[i])
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    mpc_transfers()
