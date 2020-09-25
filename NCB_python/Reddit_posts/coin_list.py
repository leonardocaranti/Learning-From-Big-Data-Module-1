import pandas as pd 

coin_list_file = pd.read_csv('Reddit_posts/reddit_topic_names.txt', sep=',', index_col=False, header=None, engine='python', encoding='utf-8')
coin_list_file[1] = coin_list_file[1].str.strip()

coin_symbols = coin_list_file[0].tolist()
coin_names = coin_list_file[1].dropna().tolist()


def get_symbol(word):
    if word in coin_symbols:
        return word
    elif word in coin_names:
        return coin_list_file[0][coin_list_file[1] == word].to_numpy()[0]
    else:
        return None


