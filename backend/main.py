import fasttext
import os
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import json
import numpy as np

import google.generativeai as genai


import spacy
from collections import Counter
from wordcloud import WordCloud
import io
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import base64

from sklearn.cluster import MiniBatchKMeans
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import uvicorn

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware before defining routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#----------------------Helper Functions-----------------------

def Load_files(Dress_file, Clothes_file):
    '''
    Parameters
    ----------
    Dress_file : string
    Clothes_file : string
    The paths of both files

    Returns
    -------
    Dataframe and metadata

    '''
    
    with open(Dress_file, 'r', encoding='utf-8') as f:
        Dress = json.load(f)
    with open(Clothes_file, 'r', encoding='utf-8') as f:
        clothes = json.load(f)
        
    cloth_meta = clothes["metadata"]
    Dress_meta = Dress["metadata"]
    cloth_df = clothes["data"]
    Dress_df = Dress["data"]
    
    return cloth_meta, cloth_df, Dress_meta, Dress_df


def filter_clothes(department="All" ):

    clothDF = pd.DataFrame(cloth_df)
    
    if department == "All":
        return clothDF
    else:
        Filter = clothDF["Department Name"] == department
        return clothDF[Filter]


def filter_Dress(brand):

    DressDF = pd.DataFrame(Dress_df)
   
    Filter = (DressDF["Brand"] == brand)
    
    return DressDF[Filter]



def sentiment_breakdown(dataframe):
    
    tot = dataframe['label'].count()

    
    positive = dataframe[dataframe['label']==1]
    neutral = dataframe[dataframe['label']==0]
    negative = dataframe[dataframe['label']==-1]

    
    positive_percent = float(list(positive.count()/tot)[0])
    neutral_percent = float(list(neutral.count()/tot)[0])
    negative_percent = float(list(negative.count()/tot)[0])
    
    return {"data":[positive, neutral, negative], "percentage": [positive_percent, neutral_percent, negative_percent]}
    

# Abandon!! TOOOOOOOO SLOWWWWWWWWW
def keyword_analysis(positive, neutral, negative):
    """
    Return: Overall Top 10, Negative Top 10, Word_Clouds (Overall and negative)
    Here we use Spacy because we only want nouns as the keywords
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    


    pos_count = Counter()
    neu_count = Counter()
    neg_count = Counter()

    # positive
    pos_docs = nlp.pipe(list(positive["Text"]), batch_size=200, n_process=-1)  # Use multiple processes for speed

    for doc in pos_docs:
        nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
        pos_count.update(nouns)

    # neutral
    neu_docs = nlp.pipe(list(neutral["Text"]), batch_size=200, n_process=-1)  # Use multiple processes for speed

    for doc in neu_docs:
        nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
        neu_count.update(nouns)
        
    # negative
    neg_docs = nlp.pipe(list(negative["Text"]), batch_size=200, n_process=-1)  # Use multiple processes for speed

    for doc in neg_docs:
        nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
        neg_count.update(nouns)
                         
        
    # Top 10 keywords
    tot_counter = pos_count + neg_count + neu_count
    Overall_top10 = tot_counter.most_common(10)
    Negative_top10 = neg_count.most_common(10)
    
    WC_bytes = {}
    # Generate word clouds + save as byte streams
    if tot_counter:
       
        wordcloud = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(tot_counter)
        # To bytes
        img_byte_array = io.BytesIO()
        wordcloud.to_image().save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        WC_bytes["overall"] = img_byte_array
    
    if neg_count:
       
        wordcloud = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(neg_count)
        # To bytes
        img_byte_array = io.BytesIO()
        wordcloud.to_image().save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        WC_bytes["neg"] = img_byte_array

    return Overall_top10, Negative_top10, WC_bytes



def keyword_analysis_fast(positive, neutral, negative):
    stop_words = set(stopwords.words("english"))
    neutral_clothing_terms = {"wedding","party","office","casual","formal","vacation","trendy","classic","vintage","modern","colorful","plain",
                              "pattern","solid","oversized","snug","loose","stretchy","tailored","daily","travel","workout","lounge","outdoor","indoor",
                              "synthetic","jacket","dress","jeans","blouse","skirt","hoodie","buttons","zippers","straps","pockets","shirt", "shirts"
                              "wrinkle-free","delicate","dry-clean","black","white","floral","polka dots","stripes","medium","large","petite","plus-size", "jean", "sweater"}
    neutral_adj_and_verbs = {
    "great", "perfect", "amazing", "beautiful", "nice", "good", "bad", "awesome", "cool",
    "okay", "fine", "best", "better", "favorite", "poor", "decent", "look", "would", "is",
    "are", "was", "were", "be", "have", "had", "love", "like", "want", "get", "go", "try",
    "buy", "keep", "return", "need", "think", "seem", "feel", "give", "say", "make", "find",
    "back", "use", "see", "put", "take", "move", "bring", "choose", "im", "wear", "even", "really",
    "top", "bottom", "looked", "way", "order", "ordered", "much", "way", "one", "run", "bit", "bought",
    "tried", "didn't", "also", "little",  "disappointed", "well"
    }
    stop_words = stop_words.union(neutral_clothing_terms)
    stop_words = stop_words.union(neutral_adj_and_verbs)
    def extract_keywords(texts, top_n=10):
        counter = Counter()
        for text in texts:
            words = [word.lower() for word in text.split() if word.lower() not in stop_words]
            counter.update(words)
        return counter.most_common(top_n)

    # Extract keywords
    positive_keywords = extract_keywords(list(positive["Text"]))
    neutral_keywords = extract_keywords(list(neutral["Text"]))
    negative_keywords = extract_keywords(list(negative["Text"]))

    # Combine results
    neg_count = Counter(dict(negative_keywords))
    overall_counter = Counter(dict(positive_keywords)) + Counter(dict(neutral_keywords)) + Counter(dict(negative_keywords))
    overall_keywords = overall_counter.most_common(15)

    WC_bytes_encoded = {}

    if overall_counter:
        wordcloud = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(overall_counter)
        img_byte_array = io.BytesIO()
        wordcloud.to_image().save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        WC_bytes_encoded["overall"] = base64.b64encode(img_byte_array.read()).decode("utf-8")

    if neg_count:
        wordcloud = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(neg_count)
        img_byte_array = io.BytesIO()
        wordcloud.to_image().save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        WC_bytes_encoded["neg"] = base64.b64encode(img_byte_array.read()).decode("utf-8")
        
    overall_keywords = [word for word, _ in overall_counter.most_common(15)]
    negative_keywords = [word for word, _ in negative_keywords]
    
    return overall_keywords, negative_keywords, WC_bytes_encoded



def kmeans_clustering(dataframe, n_clusters=7):
    """
    Recommended k: the number of brands
    """
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    X = dataframe[['Price', 'Rating']]
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_



def rating_price_scatter(All_items, Selected_Brand, All_stats, n_clusters=4):
    """
    Create a scatter plot for Rating vs Price with KMeans cluster centers.

    Parameters:
        All_items (pd.DataFrame): Dataframe containing all items.
        Selected_Brand (pd.DataFrame): Dataframe containing selected brand items.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        str: Base64 encoded string of the plot image.
    """
    # Create a scatter plot for Rating vs Price
    plt.figure(figsize=(10, 6))

    plt.scatter(All_items['Price'], All_items['Rating'], color = "#B0C4DE", alpha=0.1, s=1, label="All Items")
    plt.scatter(Selected_Brand['Price'], Selected_Brand['Rating'], color = "#FC9483", alpha=0.2, s=2, label=f"{Selected_Brand['Brand'][0]}")


    # Add average price and rating for each brand
    for _, row in All_stats.iterrows():
        avg_price = row['Price']
        avg_rating = row['Rating']
        brand_name = row['Brand']
        color = '#005f69' if brand_name not in Selected_Brand['Brand'].unique() else '#082567'
        s = 8 if brand_name not in Selected_Brand['Brand'].unique() else 40
        plt.scatter(avg_price, avg_rating, color=color, s=s, marker='o')
        plt.text(avg_price, avg_rating - 0.1, brand_name, fontsize=10, ha='center', va='center', color=color)    


    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='All Items', markerfacecolor="#B0C4DE", markersize=10, alpha=0.8),
        Line2D([0], [0], marker='o', color='w', label=f"Selected Brand: {Selected_Brand['Brand'][0]}", markerfacecolor="#FC9483", markersize=10, alpha=0.8),
        ] 
    plt.legend(handles=legend_elements, title="Legend", fontsize=9, loc='best')

    # Add labels
    plt.title("Rating vs Price", fontsize=12)
    plt.xlabel("Price (USD)", fontsize=12)
    plt.ylabel("Rating (Min: 1, Max: 5)", fontsize=12)
    #plt.legend(title="Legend", fontsize=12)
    plt.grid(alpha=0.3)
    plt.ylim(1, 5)
    plt.tight_layout()

    # Save plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)

    # Encode the image to base64
    encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return encoded_image


def Range_Analysis(DF):
    """
    Generate boxplots for price and rating distributions by brand and return the images as base64-encoded strings.
    """
    results = {}

    
    plt.figure(figsize=(12, 6))
    boxprops = dict(facecolor='#A3C1AD', color='green')
    medianprops = dict(color='#008E97', markersize=5)
    whiskerprops = dict(color='#005f69')
    capprops = dict(color='green')
    flierprops = dict(marker='o', color='#8DA399', alpha=0.3, markersize=2) 

    DF.boxplot(column='Price', by='Brand', grid=False, patch_artist=True,
               boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops,
               flierprops=flierprops)

    plt.title("Price by Brand", fontsize=14)
    plt.suptitle("")
    plt.xlabel("Brand", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    results['price'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Plot for Rating
    plt.figure(figsize=(12, 6))
    DF.boxplot(column='Rating', by='Brand', grid=False, patch_artist=True,
               boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops,
               flierprops=flierprops)

    plt.title("Rating by Brand", fontsize=14)
    plt.suptitle("")  # 
    plt.xlabel("Brand", fontsize=12)
    plt.ylabel("Rating", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    results['rating'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return results


#-------------------------API CALLS---------------------------


# Load the trained FastText model during app initialization
MODEL_PATH = r"model\clothes_review_sentiment.ftz"
fasttext_model = fasttext.load_model(MODEL_PATH)
DressFILE = r"data_for_webapp\dresses.json"
FASHIONFILE = r"data_for_webapp\amazon_fashion.json"



@app.on_event("startup")
async def ini():
    """
    Initialization
    """
    
    global client

    with open(r"D:\api_key.txt", "r") as file:
        api_key = file.read().strip()
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel("gemini-1.5-flash", system_instruction="You are a data analyst.")
    
    global cloth_meta, cloth_df, Dress_meta, Dress_df
    
    cloth_meta, cloth_df, Dress_meta, Dress_df = Load_files(
        Clothes_file=FASHIONFILE, Dress_file=DressFILE)

    cloth_frame = pd.DataFrame(cloth_df)
    corpus_for_model = list(cloth_frame['Text'])

    predicted_labels = fasttext_model.predict(corpus_for_model, k=1)
    
    labels = np.array(predicted_labels) 
    labels = labels[0,:]
    
    label_vec = []
    for l in labels:
        logic = l[0].strip('__label__')
        logic = eval(logic)
        label_vec.append(logic)
        
    cloth_frame['label'] = label_vec
    cloth_df = cloth_frame.to_dict()
    

@app.get("/")
async def root():
    """
    Simple confirmation endpoint.
    """
    return {"message": "Application initialized and ready to use."}
    
    
    
    #return {'cloth_info': cloth_meta, "cloth_data": cloth_df, 
    #       "Dress_info": Dress_meta, "Dress_data": Dress_df}

@app.get("/sentiment_analysis/")
async def sentiment_analysis(department: str = Query(
        "All",
        description="Filter by department"
        )
        ):
    """
    Perform Sentiment Analysis based on queries
    """

    filtered_data = filter_clothes(department)
    
    # Task 1: Sentiment Break Down 
    response = sentiment_breakdown(filtered_data)
    pos, neu, neg = response["data"] # dataframe
    posp, neup, negp = response["percentage"] # percentage
    

 
    
    # Key phrases and themes
    overall_keywords, negative_keywords, WC_bytes = keyword_analysis_fast(pos, neu, neg)
    
    prompt_overall = (f"{overall_keywords} are top keywords in amazon fashion/clothes reviews. Summarize" 
              " and interpret what customers mainly talks about in bullet points (150 words)")
    
    prompt_neg = (f"{negative_keywords} are top keywords in negative amazon fashion/clothes reviews. Summarize" 
              " and interpret in bullet points about the major pain points (150 words)")
    
    response_overall = str(client.generate_content(prompt_overall).text)
    
    response_neg = str(client.generate_content(prompt_neg).text)
    
    return {"positive_percentage": posp, "neutral_percentage": neup, "negative_percentage": negp,
            "Overall": {"keywords": overall_keywords, "visualize": f"data:image/png;base64,{WC_bytes.get('overall')}", "interpret": response_overall},
            "Negative": {"keywords": negative_keywords, "visualize": f"data:image/png;base64,{WC_bytes.get('neg')}", "interpret": response_neg}}
    
    

@app.get("/competitive_analysis/")
async def competitive_analysis(brand: str = Query(
        "Alexander McQueen",
        description="Filter by brand"  
    )
   ):
    
    """
    Perform Competitive Analysis based on queries
    1. Scatter plot visualization: Compare price points and ratings across competitors.
    2.Current product's position: Show the product’s position relative to competitors in terms
        of price and rating.
    3. Price range analysis: Display distribution and clustering of price ranges.
    4. Ratings analysis: Highlight average ratings and total number of competitors.

    """
    All_items = pd.DataFrame(Dress_df)
    Selected_Brand = filter_Dress(brand)
    All_stats = pd.DataFrame(Dress_meta["stats"])
    
    # All images in bytes (encoded)
    scatter_image = rating_price_scatter(All_items, Selected_Brand, All_stats)
    
    range_images = Range_Analysis(All_items)
    price_range_image = range_images["price"]
    rating_range_image = range_images["rating"]
    
    stats = All_stats[["Brand", "Price","Rating", "Price Var", "Rating Var"]]
    brand = stats[stats["Brand"] == brand]

    prompt = (f"We are analyzing {brand.to_dict()} among {stats.to_dict()}"
              " Return full sentences by filling in the blanks:\n")
    
    prompt1 = prompt + (f"{brand['Brand']}is generally within a (high/median/low)"
                        " price range. The closest brand in terms of price is ()."
                        " (brand name) has (number: only count the ones whose average price is close) major competitors within similar price range."
                        " Its overall rating (number) is top (percentile) among all brands, and "
                        "is (higher than/lower than/similar to) major competitor within the same price range."
                        "is (higherthan/lower than/similar to) major competitor within the same price range."
                        "According to the rating, its consistency in quality is relatively (high/medium/low)")
    
    response = client.generate_content(prompt1).text
    
    return {"Scatter":f"data:image/png;base64,{scatter_image}", "price_image": f"data:image/png;base64,{price_range_image}", 
            "rating_image": f"data:image/png;base64,{rating_range_image}", "interpretation": response}


if __name__ == "__main__":
    
    
    # Start the FastAPI application
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",  
        port=8000,  
        reload=True  # Automatically reload the app when the code changes (useful for development)
    )
    



