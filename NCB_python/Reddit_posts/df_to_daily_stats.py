import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pos_given_up():
    only_up_df, only_down_df = df_file[df_file['P(up)'] >= 0.5], df_file[df_file['P(up)'] < 0.5]
    only_up_df_pos, only_down_df_pos = only_up_df['P(pos)'], only_down_df['P(pos)']
    print(f'Only up df: \nP(pos|up) = {np.mean(only_up_df_pos.to_numpy())} \nStd = {np.std(only_up_df_pos.to_numpy())} \n')
    print(f'Only down df: \nP(pos|down) = {np.mean(only_down_df_pos.to_numpy())} \nStd = {np.std(only_down_df_pos.to_numpy())} \n')

def plotter(col1_name, col2_name):

    r2 = r2_score(df_file[col1_name].to_numpy(), df_file[col2_name].to_numpy())

    corr = df_file[col1_name].corr(df_file[col2_name])
    print_name = col1_name + '-' + col2_name
    print(f'R2 score for {print_name}: {r2}')
    print(f'Correlation for {print_name}: {corr} \n')

    plt.scatter(df_file[col2_name].to_numpy(), df_file[col1_name].to_numpy(), color ='k')
    plt.title(f'{col1_name} - {col2_name} distribution (corr = {round(corr,5)})')
    plt.xlabel(col2_name)
    plt.ylabel(col1_name)
  
def plotter_3d(col1_name, col2_name, col3_name, ax, fig):
    ax.scatter(df_file[col2_name].to_numpy(), df_file[col1_name].to_numpy(), df_file[col3_name].to_numpy(), c='black')
    ax.set_xlabel(col2_name)
    ax.set_ylabel(col1_name)
    ax.set_zlabel(col3_name)
    ax.set(title = f'{col1_name} - {col2_name} - {col3_name} distribution')
    # fig.colorbar(pl)

def general_class_overview():
    # 2D plots
    plt.subplot(221)
    plotter('P(up)', 'P(pos)')
    plt.subplot(222)
    plotter('P(up)', 'Pos sent score')
    plt.subplot(223)
    plotter('Pos sent score', 'Neg sent score')
    plt.subplot(224)
    plotter('P(pos)', 'Pos sent score')
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = Axes3D(fig)
    plotter_3d('Pos sent score', 'Neg sent score', 'P(up)', ax, fig)
    plt.show()

def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

def get_currencies_df(df_file, num_max_symbols):
    num_rows, num_cols = df_file.shape
    stats_df = pd.DataFrame(columns = ['symbol', '# comments', 'mean P(up)', 'mean P(pos)','mean P(neg)', 'mean pos score', 'mean neg score'])

    for idx_symbol_col in range(num_max_symbols):
        symbol_col_num = num_cols - num_max_symbols + idx_symbol_col
        single_symb_df = df_file[['P(up)', 'P(pos)', 'P(neg)', 'Pos sent score', 'Neg sent score', str(symbol_col_num)]].copy()

        # Returns tuple with (col0, col1, col2, col3, col4)
        for row in single_symb_df.itertuples(index=False):
            symbol = row[5]
            # nan is a float: check is symbol is nan using its float property
            if type(symbol) == float:
                symbol = 'no_symbol'
            new_row = pd.DataFrame([[symbol, 1, row[0], row[1], row[2], row[3], np.absolute(row[4])]], 
                columns = ['symbol', '# comments', 'mean P(up)', 'mean P(pos)', 'mean P(neg)', 'mean pos score', 'mean neg score'])
            stats_df = stats_df.append(new_row, ignore_index = True)
    
    functions_dict = {'# comments':'sum', 'mean P(up)':'mean','mean P(pos)':'mean','mean P(neg)':'mean','mean pos score':'mean','mean neg score':'mean'}
    stats_df = stats_df.groupby(by='symbol').agg(functions_dict).sort_values(by='# comments', ascending=False)

    # Append final 'total' row
    final_row = [[stats_df['# comments'].sum(), wavg(stats_df, 'mean P(up)', '# comments'), wavg(stats_df, 'mean P(pos)', '# comments'), 
        wavg(stats_df, 'mean P(neg)', '# comments'), wavg(stats_df, 'mean pos score', '# comments'), wavg(stats_df, 'mean neg score', '# comments')]]
    final_row_df = pd.DataFrame(final_row, index=['total'], columns = ['# comments', 'mean P(up)', 'mean P(pos)', 
        'mean P(neg)', 'mean pos score', 'mean neg score'])
    stats_df = stats_df.append(final_row_df)
    return stats_df




if __name__ == "__main__":

    from os import listdir
    from os.path import isfile, join
    folder = 'Daily_dataframes/'
    files_list = [f for f in listdir(folder) if isfile(join(folder, f))]


    for filename in files_list:
        print(filename)
        date = filename[:-4]
        dataframe_filename = folder + date + '.csv'
        savefile_name = date + '_stats.csv'
        df_file = pd.read_csv(dataframe_filename, sep=',', header=0, engine='python',encoding='utf-8-sig')
        num_max_symbols = df_file.shape[1] - 5

        # general_class_overview()
        stats_df = get_currencies_df(df_file, num_max_symbols)
        stats_df.to_csv('Daily_stats/' + savefile_name, index=True,encoding='utf-8-sig')


