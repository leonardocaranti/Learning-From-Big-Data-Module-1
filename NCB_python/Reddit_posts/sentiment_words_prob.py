import pandas as pd 
import numpy as np

def scaler(dataset, maxim, mean):
    """ scales for 0.5-1 distribution with mean 0.75 """
    new_width, new_mean = 1/2, 3/4
    dat = dataset.sub(mean).mul( new_width / maxim )
    return dat.add(1/2 - dat.min())



def sentiwords_extraction():
    global senti_probs_df, sentiwords_df
    # Sentiwords Dataframe preparation
    sentiwords_df = pd.read_csv('Reddit_posts/SentiWords_1.1.txt', skiprows=26, sep='\t', header=None, engine='python')
    sentiwords_df = sentiwords_df.rename(columns={0: 'Word', 1: 'Sentiment'})

    chars_to_delete = [r'#a', r'#n', r'#v', r'#r']
    for char in chars_to_delete:
        sentiwords_df['Word'] = sentiwords_df['Word'].str.replace(char, '')


    thresh_sub = 0.1

    # Probabilities preparation
    senti_probs_df = sentiwords_df.copy()
    senti_probs_df['P(word|pos)'] = senti_probs_df['Sentiment'].copy()
    senti_probs_df['P(word|pos)'][senti_probs_df['Sentiment'] < 0] = np.NaN

    max_pos, mean_pos = senti_probs_df['P(word|pos)'].max(), senti_probs_df['P(word|pos)'].mean()
    print(max_pos, mean_pos, '\n')
    senti_probs_df['P(word|pos)'] = scaler(senti_probs_df['P(word|pos)'], max_pos, mean_pos)

    senti_probs_df['P(word|pos)'][senti_probs_df['Sentiment'] < 0] = 1/2 - (senti_probs_df['P(word|pos)'].mean() - 1/2) - 0.35

    

    senti_probs_df['P(word|neg)'] = senti_probs_df['Sentiment'].copy()
    senti_probs_df['P(word|neg)'] = senti_probs_df['P(word|neg)'].abs()
    senti_probs_df['P(word|neg)'][senti_probs_df['Sentiment'] > 0] = np.NaN

    max_neg, mean_neg = senti_probs_df['P(word|neg)'].max(), senti_probs_df['P(word|neg)'].mean()
    print(max_neg, mean_neg, '\n')
    senti_probs_df['P(word|neg)'] = scaler(senti_probs_df['P(word|neg)'], max_neg, mean_neg)

    senti_probs_df['P(word|neg)'][senti_probs_df['Sentiment'] > 0] = 1/2 - (senti_probs_df['P(word|neg)'].mean() - 1/2) - 0.014



if __name__ == "__main__":
    """ Run main program only if this file is executed alone, not imported  """

    sentiwords_extraction()

    print(senti_probs_df)

    sentiwords_df.to_csv('Reddit_posts/Word_sentiment_sentiwords.csv', index=False,encoding='utf-8-sig')
    senti_probs_df.to_csv('Reddit_posts/Probabilities_sentiment.csv', index=False,encoding='utf-8-sig')
