import logging
import json
import os
from datetime import datetime, timedelta
import pytz
import azure.functions as func
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
from azure.eventhub import EventHubProducerClient, EventData
import requests
import time
from azure.storage.blob import BlobServiceClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import math
from dateutil import parser

app = func.FunctionApp()
# Configure logging
logging.basicConfig(level=logging.INFO)

# Event Hub configuration
EVENT_HUB_CONN_STR = os.getenv("EVENT_HUB_CONNECTION_STRING")
EVENT_HUB_NAME = "team12eventhub"

# Stock configuration
TICKER = "NVDA"
TIMEZONE = "America/New_York"
api_key = os.getenv("TWEET_API")
storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')

connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container = "team12blobcontainer"

def send_to_eventhub(combined_data: dict) -> bool:
    """Send data to Azure Event Hub with proper error handling"""
    try:
        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENT_HUB_CONN_STR,
            eventhub_name=EVENT_HUB_NAME
        )
        
        with producer:
            event_data_batch = producer.create_batch()
            event_data_batch.add(EventData(json.dumps(combined_data)))
            producer.send_batch(event_data_batch)
        return True
        
    except Exception as e:
        logging.error(f"Failed to send data to Event Hub: {str(e)}")
        return False
    
def is_market_open(date: datetime) -> bool:
    """Check if the given date was a trading day"""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.valid_days(
        start_date=date.strftime('%Y-%m-%d'),
        end_date=date.strftime('%Y-%m-%d')
    )
    return len(schedule) > 0

def get_stock_data(ticker: str, date: datetime) -> dict:
    """Fetch stock data for a specific date"""
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        if not data.empty:
            row = data.iloc[0]
            return {
                "match_date": start_date,
                "current_open": float(row["Open"]),
                "current_high": float(row["High"]),
                "current_low": float(row["Low"]),
                "current_close": float(row["Close"]),
                "current_volume": int(row["Volume"]),
            }
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
    return None

def fetch_tweets(api_key: str, max_tweets: int, date: datetime):
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "twitter241.p.rapidapi.com"
    }

    since = date.strftime('%Y-%m-%d')
    until = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    base_url = "https://twitter241.p.rapidapi.com/search-v2"
    query_base = f"(%23%24NVDA) lang:en until:{until} since:{since}"
    
    all_tweets = []

    cursor = None

    while len(all_tweets) < max_tweets:
        params = {
            "type": "Top",
            "count": "20", 
            "query": query_base
        }
        if cursor:
            params["cursor"] = cursor
        try:
            response = requests.get(base_url, headers=headers, params=params)
            data = response.json()
            entries = data.get("result", {}).get("timeline", {}).get("instructions", [])[0].get("entries", [])
            new_tweets = []
            for entry in entries:
                item = entry.get("content", {}).get("itemContent", {})
                if item.get("itemType") == "TimelineTweet":
                    tweet_result = item.get("tweet_results", {}).get("result", {})
                    full_text = tweet_result.get("legacy", {}).get("full_text")
                    rest_id = tweet_result.get("rest_id")
                    user_id = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("id")
                    is_blue_verified = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("is_blue_verified")
                    account_created_at = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("created_at")
                    followers_count = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("followers_count")
                    friends_count = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("friends_count")
                    account_favourites_count = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("favourites_count")
                    listed_count = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("listed_count")
                    media_count = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("media_count")
                    account_possibly_sensitive = tweet_result.get("core",{}).get("user_results", {}).get("result", {}).get("legacy",{}).get("possibly_sensitive")
                    tweet_created_at = tweet_result.get("legacy", {}).get("created_at")
                    view_count = tweet_result.get("views",{}).get("count")
                    retweet_count = tweet_result.get("legacy", {}).get("retweet_count")
                    reply_count = tweet_result.get("legacy", {}).get("reply_count")
                    quote_count = tweet_result.get("legacy", {}).get("quote_count")
                    favorite_count = tweet_result.get("legacy", {}).get("favorite_count")
                    tweet_possibly_sensitive = tweet_result.get("legacy", {}).get("possibly_sensitive")
                    if full_text and rest_id:
                        clean_tweet = {
                            "rest_id": rest_id,
                            "full_text": full_text,
                            "user_id": user_id,
                            "is_blue_verified": is_blue_verified,
                            "account_created_at": account_created_at,
                            "followers_count": followers_count,
                            "friends_count": friends_count,
                            "account_favourites_count": account_favourites_count,
                            "listed_count": listed_count,
                            "media_count": media_count,
                            "account_possibly_sensitive": account_possibly_sensitive,
                            "tweet_created_at": tweet_created_at,
                            "view_count": view_count,
                            "retweet_count": retweet_count,
                            "reply_count": reply_count,
                            "quote_count": quote_count,
                            "favorite_count": favorite_count,
                            "tweet_possibly_sensitive": tweet_possibly_sensitive
                        }
                        new_tweets.append(clean_tweet)
            all_tweets.extend(new_tweets)
            if not new_tweets or len(entries) < 3:
                break
            for entry in entries:
                content = entry.get("content", {})
                if content.get("cursorType") == "Bottom":
                    cursor = content.get("value")
                    break
            time.sleep(1)  # To prevent rate limiting
        except Exception as e:
            print("Error fetching tweets:", e)
            break
    return all_tweets[:max_tweets]

def tweets_preprocessing(tweets: list) -> list:
    preprocessed_tweets = []
    if not tweets:
        return preprocessed_tweets
    for tweet in tweets:
        if tweet is None:
            continue
        try:
            new_tweet = tweet.copy()
            new_tweet["account_created_at"] = datetime.strptime(tweet["account_created_at"], "%a %b %d %H:%M:%S %z %Y").isoformat()
            dt = datetime.strptime(tweet["tweet_created_at"], '%a %b %d %H:%M:%S %z %Y')
            new_tweet["tweet_created_at"] = dt.isoformat()
            new_tweet["tweet_created_at_date"] = dt.date().isoformat()
            new_tweet["tweet_created_at_time"] = dt.strftime('%H:%M:%S')
            preprocessed_tweets.append(new_tweet)
        except (ValueError, TypeError) as e:
            continue
    return preprocessed_tweets

def clean_text(text):
    if text:
        text = re.sub(r'https?://\S+', '', text)  # Remove any URL
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    return None

def sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    if text:
        score = sia.polarity_scores(text)["compound"]
        return score
    return None

def gold_layer_processing(stock_data: dict, tweets: list) -> dict:
    if not stock_data or not tweets:
        return None
    tweets = tweets_preprocessing(tweets)
    nltk.download('vader_lexicon')
    agg_tweets = []
    
    for tweet in tweets:
        if tweet is None:
            continue
        tweet["clean_text"] = clean_text(tweet["full_text"])
        tweet["sentiment_score"] = sentiment_score(tweet["clean_text"])
        tweet["interaction_score"] = int(tweet["view_count"]) * 0.1 + tweet["retweet_count"] * 2 + tweet["reply_count"] * 1.5 + tweet["quote_count"] * 1.2 + tweet["favorite_count"] * 1.0
        tweet["favorite_ratio"] = tweet["favorite_count"] / (float(tweet["view_count"]) + 1)
        tweet["reply_ratio"] = tweet["reply_count"] / (tweet["retweet_count"] + 1)
        dt = parser.parse(tweet["account_created_at"]) 
        local_now = datetime.now(pytz.timezone(TIMEZONE))
        account_created = dt.astimezone(pytz.timezone(TIMEZONE))
        tweet["account_age_days"] = (local_now - account_created).days
        tweet["credibility_score"] = tweet["is_blue_verified"] * 2 + (math.log(tweet["followers_count"]) if tweet["followers_count"] > 0 else 0) + tweet["account_age_days"]/365
        tweet["follower_activity_score"] = (tweet["account_favourites_count"] + tweet["media_count"] + tweet["listed_count"]) / (tweet["account_age_days"] + 1)
        tweet["tweet_created_at_date"] = parser.parse(tweet["tweet_created_at_date"]).date()
        tweet["account_created_at"] = parser.parse(tweet["account_created_at"])

        tweet["day_of_week"] = tweet["tweet_created_at_date"].strftime("%A")
        tweet["is_viral"] = tweet["interaction_score"] > 2000
        tweet["is_new_account"] = (
            (tweet["tweet_created_at_date"] - tweet["account_created_at"].date()).days < 365
        )
        tweet["is_influencer"] = (
            tweet["followers_count"] > 130000 or tweet["is_blue_verified"] is True
        )
        tweet["view_count"] = tweet["view_count"] if tweet["view_count"] else 0
        tweet["interaction_score"] = tweet["interaction_score"] if tweet["interaction_score"] else 0
        tweet["favorite_ratio"] = tweet["favorite_ratio"] if tweet["favorite_ratio"] else 0
        tweet["tweet_possibly_sensitive"] = tweet["tweet_possibly_sensitive"] if tweet["tweet_possibly_sensitive"] else False
        tweet["current_open"] = stock_data["current_open"]
        tweet["current_high"] = stock_data["current_high"]
        tweet["current_low"] = stock_data["current_low"]
        tweet["current_close"] = stock_data["current_close"]
        tweet["current_volume"] = stock_data["current_volume"]
        agg_tweets.append(tweet)
        tweet["account_created_at"] = tweet["account_created_at"].strftime("%Y-%m-%d %H:%M:%S")
        tweet["tweet_created_at_date"] = tweet["tweet_created_at_date"].strftime("%Y-%m-%d")
    return agg_tweets

def save_to_blob(data: dict, date: datetime) -> bool:
    """Save data to Azure Blob Storage"""
    try:
        date = date.strftime('%Y-%m-%d')
        blob_path = f"Medallion/Gold/Stream/{date}.json"
        data_json = json.dumps(data, indent=4)
        blob_client = blob_service_client.get_blob_client(container=container, blob=blob_path)
        blob_client.upload_blob(data_json, overwrite=True)
        return True
    except Exception as e:
        logging.error(f"Failed to save data to Blob Storage: {str(e)}")
        return False
    
@app.timer_trigger(
    schedule="0 0 9 * * *", 
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=False
)
def timer_trigger_day(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.warning('The timer is past due!')
    try:
        ny_time = datetime.now(pytz.timezone(TIMEZONE))
        check_date = ny_time - timedelta(days=1)
        if not is_market_open(check_date):
            return     
        stock_data = get_stock_data(TICKER, check_date)
        tweets = fetch_tweets(api_key, 500, check_date)
        combined_data = gold_layer_processing(stock_data, tweets)
        if combined_data:
            logging.info(f"Retrieved data for {TICKER} on {check_date.strftime('%Y-%m-%d')}")
            
            if not save_to_blob(combined_data, check_date):
                logging.error("Failed to save data to Blob Storage")
            if not send_to_eventhub(combined_data):
                logging.error("Failed to send data to Event Hub")
        else:
            logging.warning(f"No data found for {TICKER} on {check_date.strftime('%Y-%m-%d')}")   
    except Exception as e:
        logging.error(f"Error in timer trigger: {str(e)}")
        raise 
    logging.info('Timer trigger function completed')