CREATE PROCEDURE sp_DailyRefresh
AS
BEGIN
    -- FACT: Daily Tweet Summary
    TRUNCATE TABLE FactDailyTweet;
    INSERT INTO FactDailyTweet SELECT * FROM ExternalDailyTweet;

    -- FACT: Viral Tweet Log
    TRUNCATE TABLE FactViralTweetLog;
    INSERT INTO FactViralTweetLog
    SELECT 
        user_id,
        tweet_created_at_date,
        CAST(full_text AS NVARCHAR(500)),
        sentiment_score,
        interaction_score,
        followers_count,
        favorite_count,
        retweet_count,
        reply_count,
        CAST(view_count AS INT),
        credibility_score
    FROM ExternalViralTweetLog;

    -- FACT: User Type Distribution
    TRUNCATE TABLE FactUserTypeDistribution;
    INSERT INTO FactUserTypeDistribution SELECT * FROM ExternalUserTypeDistribution;

    -- FACT: Top Users by Tweet Count
    TRUNCATE TABLE FactTopUsersByTweetCount;
    INSERT INTO FactTopUsersByTweetCount SELECT * FROM ExternalTopUsersByTweetCount;

    -- FACT: Top Users by Viral Tweets
    TRUNCATE TABLE FactTopUsersByViralTweets;
    INSERT INTO FactTopUsersByViralTweets SELECT * FROM ExternalTopUsersByViralTweets;

    -- FACT: Market Summary per Day
    TRUNCATE TABLE FactMarketSummaryPerDay;
    INSERT INTO FactMarketSummaryPerDay SELECT * FROM ExternalMarketSummaryPerDay;

    TRUNCATE TABLE FactInfluencerMonthlyEngagement;
    INSERT INTO FactInfluencerMonthlyEngagement (
        year,
        month,
        year_month,
        influencer_tweet_count,
        avg_sentiment_score,
        avg_interaction,
        distinct_influencers
    )
    SELECT 
        year,
        month,
        CAST(CONCAT(year, '-', RIGHT('0' + CAST(month AS VARCHAR(2)), 2), '-01') AS DATE) AS year_month,
        influencer_tweet_count,
        avg_sentiment_score,
        avg_interaction,
        distinct_influencers
    FROM ExternalInfluencerMonthlyEngagement;

    -- DIM: Date
    TRUNCATE TABLE DimDate;
    INSERT INTO DimDate
    SELECT DISTINCT
        tweet_created_at_date AS date,
        YEAR(tweet_created_at_date) AS Year,
        MONTH(tweet_created_at_date) AS Month,
        DATENAME(MONTH, tweet_created_at_date) AS MonthName,
        DAY(tweet_created_at_date) AS Day,
        DATEPART(WEEKDAY, tweet_created_at_date) AS DayOfWeek,
        DATENAME(WEEKDAY, tweet_created_at_date) AS DayOfWeekName
    FROM ExternalDailyTweet;

    -- DIM: User
    TRUNCATE TABLE DimUser;
    INSERT INTO DimUser
    SELECT
        user_id,
        is_blue_verified,
        account_created_at,
        account_age_days,
        is_new_account,
        is_influencer,
        followers_count,
        friends_count,
        account_favourites_count,
        listed_count,
        media_count,
        credibility_score,
        follower_activity_score
    FROM ExternalUserProfileDim;
END
