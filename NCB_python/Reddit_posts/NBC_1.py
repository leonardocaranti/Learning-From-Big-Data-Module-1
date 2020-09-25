from training_text_to_prob import * 
from sentiment_words_prob import *
from coin_list import get_symbol
from get_urls import get_urls_dates
import matplotlib.pyplot as plt
import nltk
import praw
import warnings
import numpy as np
import datetime



def get_files():
    """ Import files with probabilities and sentiments"""
    global Prob_file, Sentim_file, Sentim_prob_file, valid_file
    valid_file = pd.read_csv('Reddit_posts/validation_file.txt', sep='\t', names = ['up/down', 'pos/neg'], engine='python',encoding='utf-8-sig')
    try:
        Prob_file = pd.read_csv('Reddit_posts/Probabilities_training.csv', sep=',', header=0, engine='python',encoding='utf-8-sig')
        Sentim_prob_file = pd.read_csv('Reddit_posts/Probabilities_sentiment.csv', sep=',', header=0, engine='python',encoding='utf-8-sig')
        Sentim_file = pd.read_csv('Reddit_posts/Word_sentiment_sentiwords.csv', sep=',', header=0, engine='python',encoding='utf-8-sig')
    except FileNotFoundError:
        get_prob_df().to_csv('Reddit_posts/Probabilities_training.csv', index=False,encoding='utf-8-sig')
        Prob_file = pd.read_csv('Reddit_posts/Probabilities_training.csv', sep=',', header=0, engine='python',encoding='utf-8-sig')
        sentiwords_df.to_csv('Reddit_posts/Word_sentiment_sentiwords.csv', index=False,encoding='utf-8-sig')
        Sentim_file = pd.read_csv('Reddit_posts/Word_sentiment_sentiwords.csv', sep=',', header=0, engine='python',encoding='utf-8-sig')
        senti_probs_df.to_csv('Reddit_posts/Probabilities_sentiment.csv', index=False,encoding='utf-8-sig')
        Sentim_prob_file = pd.read_csv('Reddit_posts/Probabilities_sentiment.csv', sep=',', header=0, engine='python',encoding='utf-8-sig')
        

chars_to_delete_words = [r',', r'.', r':', r';', r'"', r"'", r'(', r')', r'-', r'â€', r'˜', r'”', r'™', r'‘', r'—', r'/'] 
def delete_unnec_symbols(comment):
    for char in chars_to_delete_words:
        comment = comment.replace(char, ' ')
    return comment


Lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_word(word):
    return Lemmatizer.lemmatize(Lemmatizer.lemmatize(Lemmatizer.lemmatize(Lemmatizer.lemmatize(Lemmatizer.lemmatize(word, pos='n'), pos='v'), pos='a'), pos='s'), pos='r')


def set_post_topic(word):
    global prob_classes, history_word_probs, count_unused_words
    # Calculate and set posteriors for topic analysis
    P_B_up = Prob_file['P(word|up)'][Prob_file['Word'] == word].to_numpy()
    P_B_down = Prob_file['P(word|down)'][Prob_file['Word'] == word].to_numpy()

    if len(P_B_up) > 0:
        old_prob_classes = prob_classes.copy()
        prob_classes[0] = P_B_up * old_prob_classes[0] / (P_B_up * old_prob_classes[0] + P_B_down * old_prob_classes[1])
        prob_classes[1] = P_B_down * old_prob_classes[1] / (P_B_up * old_prob_classes[0] + P_B_down * old_prob_classes[1])

        history_word_probs.append([P_B_up[0], P_B_down[0]])
    else:
        count_unused_words += 1

    # print(f'P_B_up {P_B_up}, P_B_down {P_B_down}')
    # print(f'New prob classes {prob_classes}')


def set_post_sent(word):
    global pos_sent_scores, neg_sent_scores
    # Add sentiment values for sentiment analysis
    single_w_sentiment = Sentim_file['Sentiment'][Sentim_file['Word'] == word].to_numpy()

    if len(single_w_sentiment) > 0:
        if len(single_w_sentiment) > 1:
            single_w_sentiment = np.array([np.mean(single_w_sentiment)])
        if single_w_sentiment[0] > 0:
            pos_sent_scores.append(single_w_sentiment[0])
        if single_w_sentiment[0] < 0:
            neg_sent_scores.append(single_w_sentiment[0])

    # print(f'Word sentiment {single_w_sentiment}')
    # print('Pos sent score:', np.mean(pos_sent_scores[1:]), 'Neg sent score:', np.mean(neg_sent_scores[1:]))


def set_post_prob_sent(word):
    global sent_classes
    P_B_pos = Sentim_prob_file['P(word|pos)'][Sentim_prob_file['Word'] == word].to_numpy()
    P_B_neg = Sentim_prob_file['P(word|neg)'][Sentim_prob_file['Word'] == word].to_numpy()
    if len(P_B_pos) > 0:
        if len(P_B_pos) > 1:
            P_B_pos, P_B_neg = np.mean(P_B_pos), np.mean(P_B_neg)


        old_sent_classes = sent_classes.copy()
        if P_B_pos != 0:
            sent_classes[0] = P_B_pos * old_sent_classes[0] / (P_B_pos * old_sent_classes[0] + (1 - P_B_pos) * (1-old_sent_classes[0]))
        if P_B_neg != 0:
            sent_classes[1] = P_B_neg * old_sent_classes[1] / (P_B_neg * old_sent_classes[1] + (1 - P_B_neg) * (1-old_sent_classes[1]))

    # print(f'P_B_pos {P_B_pos}, P_B_neg {P_B_neg}')
    # print(f'New sent classes {sent_classes} \n')

def append_for_plot():
    global history_prob_classes, history_sent_classes, word_count
    # Plotting part: append important values to be plotted later
    history_prob_classes.append(list(prob_classes))
    history_sent_prob_classes.append(list(sent_classes))
    mean_positive = np.mean( np.array(pos_sent_scores)[1:] )
    mean_negative = np.absolute(np.mean( np.array(neg_sent_scores)[1:] ))
    history_sent_classes.append([mean_positive, mean_negative])
    word_count += 1


def append_symbols(word):
    global symbols_list
    symbol = get_symbol(word)
    if symbol != None:
        symbols_list.append(symbol)


def final_sent_score():
    global pos_sent_score, neg_sent_score
    # If there are no negative/positive words, set value as 0
    if len(pos_sent_scores) > 1:
        del pos_sent_scores[0]
    if len(neg_sent_scores) > 1:
        del neg_sent_scores[0]
    pos_sent_score, neg_sent_score = np.mean(np.array(pos_sent_scores)), np.mean(np.array(neg_sent_scores))


def prediction_outcome_conf_mat():
    global confusion_matrix_topic, confusion_matrix_sent, confusion_matrix_avg_sent
    # Caulculating prediction outcome: 1 was saved for uptrend, 0 for downtrend. 
    # True Positive (TP) is predicted uptrend and actual uptrend. 
    if 0 < count_valid < 41:

        # Topic
        saved_class = valid_file['up/down'][valid_file.index == count_valid-1].to_numpy()[0]
        actual_class = np.absolute(prob_classes.argmax() - 1)

        if saved_class == actual_class:
            if saved_class == 1:
                confusion_matrix_topic[0,0] += 1
            else:
                confusion_matrix_topic[1,1] += 1
        else:
            if saved_class == 1:
                confusion_matrix_topic[1,0] += 1
            else:
                confusion_matrix_topic[0,1] += 1
        
        # Sentiment
        saved_class = valid_file['pos/neg'][valid_file.index == count_valid-1].to_numpy()[0]
        actual_class = np.absolute(sent_classes.argmax() - 1)

        if saved_class == actual_class:
            if saved_class == 1:
                confusion_matrix_sent[0,0] += 1
            else:
                confusion_matrix_sent[1,1] += 1
        else:
            if saved_class == 1:
                confusion_matrix_sent[1,0] += 1
            else:
                confusion_matrix_sent[0,1] += 1

        # Avg sentiment
        saved_class = valid_file['pos/neg'][valid_file.index == count_valid-1].to_numpy()[0]
        arr = np.array([pos_sent_score, neg_sent_score])
        actual_class = np.absolute(arr.argmax() - 1)
        print(f'saved:{saved_class}, actual: {actual_class} (avg sent)')

        if saved_class == actual_class:
            if saved_class == 1:
                confusion_matrix_avg_sent[0,0] += 1
            else:
                confusion_matrix_avg_sent[1,1] += 1
        else:
            if saved_class == 1:
                confusion_matrix_avg_sent[1,0] += 1
            else:
                confusion_matrix_avg_sent[0,1] += 1


def printing():
    # Display results: printing 
    print(comment)
    # corr_result = ['Incorrect','Correct'][correct_bool]
    # print(f'{output} --- {corr_result} classification ')
    print(f'Symbols recognised: {symbols_list}')
    print(f'P(up|words) = {prob_classes[0].round(6)} \nP(down|words) = {prob_classes[1].round(6)}')
    print(f'P(pos|words) = {sent_classes[0].round(6)} \nP(neg|words) = {sent_classes[1].round(6)}')
    print(f'Positive sentiment score: {pos_sent_score.round(6)} \nNegative sentiment score: {np.absolute(neg_sent_score).round(6)} \n')


def plotting():
    global history_prob_classes, history_sent_classes, history_sent_prob_classes
    # Plot graph of probabilities vs word count, replace nan's with 0's
    def remove_nan(list_):
        df = pd.DataFrame(list_).fillna(0)
        return df.values.tolist()
    history_sent_classes = np.array(remove_nan(history_sent_classes))
    history_prob_classes = np.array(history_prob_classes)
    history_sent_prob_classes = np.array(history_sent_prob_classes)

    plt.title('Text processing history')
    plt.plot(np.arange(word_count), history_prob_classes[:,0], label='P(up|words)', color='#ff640a')
    # plt.plot(np.arange(word_count), history_prob_classes[:,1], label='P(down|words)', color='#944a20')
    plt.plot(np.arange(word_count), history_sent_classes[:,0], label='Positive sentiment score', color='#0a8dff', linestyle='dashed')
    plt.plot(np.arange(word_count), history_sent_classes[:,1], label='Negative sentiment score', color='#063966', linestyle='dashed')
    plt.plot(np.arange(word_count), history_sent_prob_classes[:,0], label='P(pos|words)', color='green', linestyle='dotted')
    plt.plot(np.arange(word_count), history_sent_prob_classes[:,1], label='P(neg|words)', color='purple', linestyle='dotted')
    # plt.plot(np.arange(word_count), history_prob_classes[:,1] + history_prob_classes[:,0], label='Check', color='gray', linestyle='-.')
    plt.legend(loc='best')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='dotted', color='k', alpha=0.8)
    plt.grid(which='minor', linestyle='dotted', color='k', alpha=0.3)
    plt.xlabel('Number of non-stopwords processed')
    plt.ylabel('Probability / Score')
    plt.show()


def training_text_outline(plotting=False):
    global mean_up_train, median_up_train, std_up_train, mean_down_train, median_down_train, std_down_train
    P_up_distr, P_down_distr = Prob_file['P(word|up)'].to_numpy(), Prob_file['P(word|down)'].to_numpy()
    mean_up_train, median_up_train, std_up_train = np.mean(P_up_distr), np.median(P_up_distr), np.std(P_up_distr)
    mean_down_train, median_down_train, std_down_train = np.mean(P_down_distr), np.median(P_down_distr), np.std(P_down_distr)
    skewness_up = 3 * ( mean_up_train - median_up_train) / std_up_train
    skewness_down = 3 * ( mean_down_train - median_down_train) / std_down_train

    if plotting:
        print(f'Up-skwenwss: {skewness_up.round(5)}')
        print(f'Down-skwenwss: {skewness_down.round(5)}')

        kwargs = dict(alpha=0.5, bins=100) # density=True, stacked=True
        plt.subplot(211)
        plt.hist(P_up_distr, **kwargs, color='g', label='P(up)')
        plt.axvline(x=mean_up_train, color = 'black', ls='dotted', label='Mean')
        plt.axvline(x=median_up_train, color = 'black', ls='dashdot', label='Median')
        plt.gca().set(title='Distribution of up words probabilities', ylabel='Frequency', xlabel='Probability')
        plt.legend()

        plt.subplot(212)
        plt.hist(P_down_distr, **kwargs, color='r', label='P(down)')
        plt.axvline(x=mean_down_train, color = 'black', ls='dotted', label='Mean')
        plt.axvline(x=median_down_train, color = 'black', ls='dashdot', label='Median')
        plt.gca().set(title='Distribution of down words probabilities', ylabel='Frequency', xlabel='Probability')
        plt.legend()
        
        plt.show()


def check_comment_validity():
    global history_word_probs
    # Use history_word_probs and number of words in a comment - TO FINISH
    if len(history_word_probs) == 0:
        validity_bool = False
        return validity_bool

    history_word_probs = np.array(history_word_probs)
    mean_up_comm, mean_down_comm, num_words_up_comm = np.mean(history_word_probs[:,0]), np.mean(history_word_probs[:,1]), history_word_probs.shape[0]
    
    length_thresh, perc_thresh = 6, 40/100
    up_thresh, down_thresh = mean_up_train, mean_down_train

    check_bool = num_words_up_comm / (num_words_up_comm + count_unused_words) > perc_thresh and num_words_up_comm > length_thresh
    mean_bool = mean_up_comm > up_thresh and mean_down_comm > down_thresh

    if check_bool and mean_bool:
        validity_bool = True
        if printing_bool: 
            print(f'\t \t Comment \t Training')
            print(f'Mean_up \t {mean_up_comm.round(5)} \t {mean_up_train.round(5)}')
            print(f'Mean_down \t {mean_down_comm.round(5)} \t {mean_down_train.round(5)}')
            print(f'Length \t \t {num_words_up_comm}')
            print(f'Length perc. \t {round(num_words_up_comm / (num_words_up_comm + count_unused_words) * 100, 5)}')
    else:
        validity_bool = False
    
    return validity_bool


def append_daily_df():
    global daily_df_list
    orig_list = [prob_classes[0], sent_classes[0], sent_classes[1], pos_sent_score, neg_sent_score]
    for symbol in symbols_list:
        if symbol not in orig_list:
            orig_list.append(symbol)
    daily_df_list.append(orig_list)


# -------------------------- Main loop --------------------------
# Client ID: 
# Client secret: 
# App name: 
# Username: 
# Password: 

printing_bool = False

client_id = ''
client_secret = ''
user_agent = ''
username = ''
password = ''

if printing_bool: 
    print(f'Connecting to reddit')

reddit = praw.Reddit(client_id = client_id,
                     client_secret = client_secret,
                     user_agent = user_agent, 
                     username = username, 
                     password = password)


urls, dates = get_urls_dates()
for url, date in zip(urls, dates):
    print(date)
    print(url)
    
    submission = reddit.submission(url=url)

    submission.comments.replace_more(limit=None)

    warnings.filterwarnings("ignore") # Ignore warning of taking mean of empty array. Watch out, all warningas are ignored...
    get_files()
    training_text_outline(plotting=False)

    num_dim = 2
    confusion_matrix_topic, confusion_matrix_sent, confusion_matrix_avg_sent = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))
    count_valid, count_invalid = 0, 0
    daily_df_list = []

    if printing_bool:
        print(f'Initiating text analysis \n')

    # for comment in comments_file[0]:
    for comment in submission.comments.list():

        comment = comment.body

        prob_classes, sent_classes = np.ones((num_dim)) * 1/num_dim, np.ones((num_dim)) * 1/num_dim
        history_prob_classes, history_word_probs, history_sent_classes, history_sent_prob_classes, word_count = [], [], [], [], 0
        pos_sent_scores, neg_sent_scores, count_unused_words = [0], [0], 0
        symbols_list = []

        for word in delete_unnec_symbols(comment).lower().split():
            if word not in stopwords:

                word = lemmatize_word(word)

                set_post_topic(word)
                set_post_sent(word)
                set_post_prob_sent(word)
                append_for_plot()
                append_symbols(word)

        final_sent_score()


        if check_comment_validity():
            if printing_bool:
                printing()
                plotting()
            append_daily_df()
            print(f'Count valid: {count_valid}') # Has to get to 41
            prediction_outcome_conf_mat()
            count_valid += 1
        else:
            count_invalid += 1


        if (count_valid + count_invalid) % 50 == 0:
            print('#', end='')

        
        if count_valid > 40: 
            print(f'Confusion matrices: \n {confusion_matrix_topic} \n {confusion_matrix_sent} \n {confusion_matrix_avg_sent}')
            break

        

    # print(f'Confusion matrix: \n {confusion_matrix}')

    daily_df = pd.DataFrame(daily_df_list).rename(columns={0:'P(up)', 1: 'P(pos)', 2: 'P(neg)', 3:'Pos sent score', 4:'Neg sent score'})

    if printing_bool: 
        print('\nPredictions completed')
        print(f'{count_valid} valid comments out of {count_valid + count_invalid} ')  
        print(f'Daily dataframe of comments: \n {daily_df} \n')  

    name = date
    daily_df.to_csv('Reddit_posts/Daily_dataframes/' + name + '.csv', index=False,encoding='utf-8-sig')
    print()
