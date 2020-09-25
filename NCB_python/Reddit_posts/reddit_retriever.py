# Client ID: A4kDyOSOOa69tQ 
# Client secret: bpVkz_6J5VqSObQH1D9PWpV3hMs
# App name: Reddit Sca=raper
# Username: leonardocaranti
# Password: passuord
import praw

client_id = 'A4kDyOSOOa69tQ'
client_secret = 'bpVkz_6J5VqSObQH1D9PWpV3hMs'
user_agent = 'windows:reddit_scraper:v.1 by u/leonardocaranti'
username = 'leonardocaranti'
password = 'passuord'

reddit = praw.Reddit(client_id = client_id,
                     client_secret = client_secret,
                     user_agent = user_agent, 
                     username = username, 
                     password = password)
            
print(f'Reddit Read Only Mode Bool: {reddit.read_only}')


url = "https://www.reddit.com/r/CryptoCurrency/comments/isxi64/daily_discussion_september_15_2020_gmt0/"
submission = reddit.submission(url=url)
submission.comments.replace_more(limit=None)

warnings.filterwarnings("ignore") # Ignore warning of taking mean of empty array. Watch out, all warningas are ignored...
get_files()

num_dim = 2
confusion_matrix = np.zeros((2,2))


for comment in submission.comments.list():

    comment = comment.body

    prob_classes = np.ones((num_dim)) * 1/num_dim
    history_prob_classes, history_sent_classes, word_count = [], [], 0
    pos_sent_scores, neg_sent_scores = [0], [0]

    for word in comment.split():
        if word not in stopwords:
            
            word = lemmatize_word(word)

            set_post_topic(word)
            set_post_sent(word)
            append_for_plot()

    final_sent_score()
    # prediction_outcome_conf_mat()

    printing()
    # plotting()

print('\nPredictions completed')
# print(f'Confusion matrix: \n {confusion_matrix}')