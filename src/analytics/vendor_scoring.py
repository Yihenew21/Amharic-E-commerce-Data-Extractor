import pandas as pd
import json
import numpy as np
from datetime import datetime

# Load metadata from cleaned.parquet
df_meta = pd.read_parquet('data/processed/cleaned.parquet')

# Load NER annotations from JSON
with open('data/labeled/raw_for_annotation/messages_for_manual_labeling.json', encoding='utf-8') as f:
    ner_data = json.load(f)

# Convert NER data to DataFrame (one row per message, columns for each entity type)
rows = []
for msg in ner_data:
    row = {'message_id': msg['message_id']}
    # For each entity type, get the first occurrence (or None)
    for entity_type in ['PRODUCT', 'PRICE', 'LOCATION', 'BRAND', 'SIZE', 'CONTACT']:
        ents = [a['text'] for a in msg.get('annotations', []) if a['entity'] == entity_type]
        row[entity_type.lower()] = ents[0] if ents else None
    rows.append(row)
df_ner = pd.DataFrame(rows)

# Merge metadata and NER results on message_id
df = pd.merge(df_meta, df_ner, on='message_id', how='left')

# Convert date to datetime
if not np.issubdtype(df['date'].dtype, np.datetime64):
    df['date'] = pd.to_datetime(df['date'])

# --- Vendor Analytics ---
scorecard = []
for vendor, group in df.groupby('channel_name'):
    # Posting frequency (posts/week)
    if len(group) > 1:
        days = (group['date'].max() - group['date'].min()).days + 1
        posts_per_week = len(group) / (days / 7) if days > 0 else len(group)
    else:
        posts_per_week = len(group)
    # Average views per post
    avg_views = group['views'].mean()
    # Top performing post
    top_post = group.loc[group['views'].idxmax()]
    top_product = top_post.get('product', None)
    top_price = top_post.get('price', None)
    top_views = top_post['views']
    # Average price point (extract numeric value)
    def extract_price(val):
        if pd.isnull(val):
            return None
        # Remove commas, birr, etc.
        digits = ''.join([c for c in str(val) if c.isdigit() or c == '.'])
        try:
            return float(digits)
        except:
            return None
    prices = group['price'].dropna().map(extract_price).dropna().astype(float)
    avg_price = prices.mean() if not prices.empty else None
    # Lending score (simple weighted sum)
    lending_score = 0.5 * (avg_views if not np.isnan(avg_views) else 0) + 0.5 * posts_per_week
    scorecard.append({
        'Vendor': vendor,
        'Posts/Week': round(posts_per_week, 2),
        'Avg. Views/Post': round(avg_views, 2) if not np.isnan(avg_views) else None,
        'Avg. Price (ETB)': round(avg_price, 2) if avg_price else None,
        'Top Product': top_product,
        'Top Price': top_price,
        'Top Views': top_views,
        'Lending Score': round(lending_score, 2)
    })

scorecard_df = pd.DataFrame(scorecard)
scorecard_df = scorecard_df.sort_values('Lending Score', ascending=False)

# Output to CSV
scorecard_df.to_csv('reports/vendor_scorecard.csv', index=False)
print('Vendor scorecard saved to reports/vendor_scorecard.csv')
print(scorecard_df)
