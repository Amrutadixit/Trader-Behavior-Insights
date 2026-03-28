#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip uninstall numpy -y
pip install numpy==1.26.4


# In[ ]:


get_ipython().system('pip install --force-reinstall pandas matplotlib seaborn')


# In[ ]:


pip uninstall numpy -y
pip install numpy==1.26.4
pip install --force-reinstall pandas matplotlib seaborn


# In[1]:


import numpy as np
import pandas as pd

print(np.__version__)


# In[3]:


import os
print(os.getcwd())
print(os.listdir())


# In[4]:


print(os.listdir())


# In[5]:


['fear_greed.csv', 'historical_data.csv', ...]


# In[9]:


import pandas as pd

sentiment = pd.read_csv('fear_greed_index.csv')
trades = pd.read_csv('historical_data.csv')

print(sentiment.head())
print(trades.head())


# In[12]:


print(sentiment.columns)


# In[16]:


['date ', ' Fear/Greed']


# In[17]:


# Clean column names
sentiment.columns = sentiment.columns.str.strip()

print(sentiment.columns)


# In[18]:


sentiment['date'] = pd.to_datetime(sentiment['date'])


# In[19]:


sentiment.rename(columns={'Fear/Greed': 'sentiment'}, inplace=True)


# In[20]:


import pandas as pd

# Clean column names
sentiment.columns = sentiment.columns.str.strip()

# Convert date
sentiment['date'] = pd.to_datetime(sentiment['date'])

# Rename for clarity
sentiment.rename(columns={'Fear/Greed': 'sentiment'}, inplace=True)

print(sentiment.head())


# In[28]:


print("Sentiment columns:", sentiment.columns)
print("Trades columns:", trades.columns)


# In[29]:


sentiment.columns = sentiment.columns.str.strip().str.lower()
trades.columns = trades.columns.str.strip().str.lower()


# In[32]:


trades.columns = trades.columns.str.strip().str.lower().str.replace(" ", "_")
sentiment.columns = sentiment.columns.str.strip().str.lower()


# In[33]:


print(trades.columns)


# In[35]:


'timestamp'


# In[36]:


trades['timestamp'] = pd.to_datetime(trades['timestamp'])
trades['date'] = trades['timestamp'].dt.date


# In[37]:


sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date


# In[38]:


merged = pd.merge(trades, sentiment, on='date', how='left')


# In[40]:


print(merged.columns)


# In[41]:


'closed_pnl'


# In[42]:


merged['is_profit'] = merged['closed_pnl'] > 0


# In[43]:


merged['side'] = merged['side'].map({'BUY': 1, 'SELL': -1})


# In[44]:


# Profit flag
merged['is_profit'] = merged['closed_pnl'] > 0

# Trade direction
merged['side'] = merged['side'].map({'BUY': 1, 'SELL': -1})


# In[46]:


print(merged.columns)


# In[47]:


profit = merged.groupby('classification')['closed_pnl'].mean()
print(profit)


# In[48]:


# Rename column (optional but clean)
merged.rename(columns={'classification': 'sentiment'}, inplace=True)

# Group by sentiment
profit = merged.groupby('sentiment')['closed_pnl'].mean()

print(profit)


# In[49]:


# Win rate
merged['is_profit'] = merged['closed_pnl'] > 0
win_rate = merged.groupby('sentiment')['is_profit'].mean()

print(win_rate)


# In[51]:


size_analysis = merged.groupby('sentiment')['size_usd'].mean()
print(size_analysis)


# In[52]:


profit = merged.groupby('sentiment')['closed_pnl'].mean()


# In[53]:


merged['is_profit'] = merged['closed_pnl'] > 0
win_rate = merged.groupby('sentiment')['is_profit'].mean()


# In[54]:


side_analysis = merged.groupby(['sentiment', 'side']).size()
print(side_analysis)


# In[55]:


trade_count = merged.groupby('sentiment').size()
print(trade_count)


# In[57]:


get_ipython().system('pip install matplotlib')


# In[59]:


import matplotlib.pyplot as plt
print("Matplotlib working!")


# In[60]:


profit.plot(kind='bar', title='Average PnL by Sentiment')
plt.show()


# In[61]:


get_ipython().system('pip install seaborn')


# In[63]:


merged.columns = merged.columns.str.strip().str.lower().str.replace(' ', '_')
print(merged.columns)


# In[64]:


features = merged[['size_usd', 'closed_pnl']].fillna(0)


# In[65]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
merged['cluster'] = kmeans.fit_predict(features)

print(merged[['size_usd', 'closed_pnl', 'cluster']].head())


# In[66]:


cluster_summary = merged.groupby('cluster')[['size_usd', 'closed_pnl']].mean()
print(cluster_summary)


# In[ ]:




