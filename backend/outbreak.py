import feedparser

RSS_URL = "https://news.google.com/rss/search?q=disease+outbreak"


def get_outbreak_alerts():

    feed = feedparser.parse(RSS_URL)

    alerts = []

    for entry in feed.entries[:5]:

        alerts.append({
            "title": entry.title,
            "link": entry.link
        })

    return alerts