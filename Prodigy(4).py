import ssl
import certifi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

plt.style.use('ggplot')


ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')


df=pd.read_csv("/Users/raghavender/Datasets:Research/Reviews.csv",skiprows=1)
df=df.head(500)

sia = SentimentIntensityAnalyzer()
res={}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text= row['Text']
    myid= row['Id']
    res[myid]=sia.polarity_scores(text)

df_res=pd.DataFrame(res).T
df_res=df_res.reset_index().rename(columns={'index':'Id'})
df_res=df_res.merge(df, how='left')


fig, axs= plt.subplots(1,3, figsize=(15,5))
sns.barplot(data=df_res,x='Score',y='pos', ax=axs[0])
sns.barplot(data=df_res,x='Score',y='neg', ax=axs[1])
sns.barplot(data=df_res,x='Score',y='neu', ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Negative")
axs[2].set_title("Neutral")
plt.show()