# Stop words from http://xpo6.com/download-stop-word-list/ 

import pandas as pd 
import numpy as np
import nltk

def get_words_df(training_text, col_num = 0):
    global chars_to_delete_words
    """ Get words set with number of occurences within a training text  """

    # Initiate lemmatizer
    Lemmatizer = nltk.stem.WordNetLemmatizer()

    # Make into a pandas series with one column of lower case comments
    text = training_text[col_num].str.lower()

    # Delete useless characters
    chars_to_delete_words = [r',', r'.', r':', r';', r'"', r"'", r'(', r')', r'-', r'â€', r'˜', r'”', r'™', r'‘', r'—', r'/']
    for char in chars_to_delete_words:
        text = text.str.replace(char, '')

    # Split into words
    text = text.str.split()

    # Define word set and count words, excluding stopwords
    words_set, words_count = [], []
    for comment in text:
        for word in comment:
            if word not in stopwords and word[0].isalpha():
                word = Lemmatizer.lemmatize(Lemmatizer.lemmatize(Lemmatizer.lemmatize(Lemmatizer.lemmatize(Lemmatizer.lemmatize(word, pos='n'), pos='v'), pos='a'), pos='s'), pos='r')
                if word not in words_set:
                    words_set.append(word)
                    words_count.append(1)
                else: 
                    index = words_set.index(word)
                    words_count[index] = words_count[index] + 1

    # Reorganize dataframe, add probabilities column
    words_df = pd.DataFrame([words_set, words_count]).T.rename(columns={0: 'Word', 1: 'Frequency'})
    freq_numpy = words_df['Frequency'].to_numpy()

    prob_numpy, freq_sum = np.ones((freq_numpy.shape)), np.sum(freq_numpy)
    for i, freq in enumerate(freq_numpy):
        prob_numpy[i] = freq / freq_sum
    words_df['Probability'] = prob_numpy

    # Sort by descending frequency
    words_df = words_df.sort_values(by=['Frequency'], ascending=False, ignore_index=True)
    
    return words_df


def save_print(file, filename):
    """ Save and print the training text and associated probabilities """
    words_df = get_words_df(file)
    words_df.to_csv(filename, index=False)
    print(f'\nTraining text saved as {filename}: \n {words_df} \n \n')


def get_prob_df():

    """ Get two files to combine into one large dataframe, with columns for frequency, frequency with 
    additive smoothing, and P(B|A) probabilities, for every dimension (uptrend, downtrend).  """

    # Select files and initiate the combination of word count
    df_up = get_words_df(uptrend_tt)[['Word', 'Frequency']]
    df_down = get_words_df(downtrend_tt)[['Word', 'Frequency']]
    total_words, total_count_up = df_up['Word'].tolist(), df_up['Frequency'].tolist()
    total_count_down = list(np.zeros((df_up['Frequency'].to_numpy().shape)))

    # Loop to count the words, to add to lists of word counts
    down_freq_list, down_word_list = df_down['Frequency'].tolist(), df_down['Word'].tolist()
    for index_down, word in enumerate(down_word_list):
        if word in total_words:
            index_total = total_words.index(word)
            total_count_down[index_total] = down_freq_list[index_down]
        else:
            total_words.append(word)
            total_count_up.append(0)
            total_count_down.append(down_freq_list[index_down])
    
    # Cleaning up: making lists into a new dataframe and adding additive smoothing columns
    prob_df = pd.DataFrame([total_words, total_count_up, total_count_down]).T.rename(columns={0: 'Word', 1: 'Freq up', 2: 'Freq down'})
    prob_df['Freq up smooth'] = prob_df['Freq up'].to_numpy() + 1
    prob_df['Freq down smooth'] = prob_df['Freq down'].to_numpy() + 1
    prob_df = prob_df.sort_values(by=['Word'], ascending=True, ignore_index=True)[['Word', 'Freq up', 'Freq up smooth', 'Freq down', 'Freq down smooth']]
    
    # Adding probabilities column
    freq_up_numpy, freq_down_numpy = prob_df['Freq up smooth'].to_numpy(), prob_df['Freq down smooth'].to_numpy()
    prob_up, prob_down = np.zeros((freq_up_numpy.shape)), np.zeros((freq_down_numpy.shape))
    sum_up, sum_down = np.sum(freq_up_numpy), np.sum(freq_down_numpy)
    for i in range(len(prob_up)):
        prob_up[i] = freq_up_numpy[i] / sum_up
        prob_down[i] = freq_down_numpy[i] / sum_down
    prob_df['P(word|up)'] = prob_up
    prob_df['P(word|down)'] = prob_down

    return prob_df


# File importing
comments_file = pd.read_csv('Reddit_posts/comments.txt', sep='///', header=None, engine='python',encoding='utf-8')
uptrend_tt = pd.read_csv('Reddit_posts/uptrend_training_text_news.txt', sep='\n', header=None, engine='python',encoding='utf-8')
downtrend_tt = pd.read_csv('Reddit_posts/downtrend_training_text_news.txt', sep='\n', header=None, engine='python',encoding='utf-8')
stopwords = pd.read_csv('Reddit_posts/stop-word_list.txt', sep=',', header=None, engine='python',encoding='utf-8')
stopwords = stopwords.iloc[0].str.strip().values.tolist()


if __name__ == "__main__":
    """ Run main program only if this file is executed alone, not imported  """
    get_prob_df().to_csv('Reddit_posts/Probabilities_training.csv', index=False,encoding='utf-8-sig')
