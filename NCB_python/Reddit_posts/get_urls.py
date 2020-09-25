import datetime
import praw

def get_date(submission):
	time = submission.created
	return datetime.date.fromtimestamp(time)

def get_urls_dates():
    
    client_id = ''
    client_secret = ''
    user_agent = ''
    username = ''
    password = ''
    reddit = praw.Reddit(client_id = client_id,
                        client_secret = client_secret,
                        user_agent = user_agent, 
                        username = username, 
                        password = password)



    main_url = 'https://www.reddit.com/r/CryptoCurrency/?f=flair_name%3A"OFFICIAL"'
    subreddit = reddit.subreddit('CryptoCurrency')
    urls, dates = [], []

    for submission in subreddit.search('flair:"'+'OFFICIAL'+'"', sort='new', syntax='lucene', limit=999):
        date = str(get_date(submission))
        year = date[0:4]
        month = date[5:7]
        day = date[8:10]

        
        validity_bool = False

        if month == '08':
            if day < '23':
                validity_bool = True

        elif month == '07':
            validity_bool = True


        if validity_bool:
            new_date = day + '_' + month + '_' + year
            dates.append(new_date)
            urls.append(submission.url)

        if month == '06':
            break
    
    return urls, dates
