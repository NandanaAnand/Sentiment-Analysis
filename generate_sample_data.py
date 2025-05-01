# generate_sample_data.py
import pandas as pd
import numpy as np
import random

# Example product categories and adjectives for generating realistic reviews
PRODUCTS = ['phone', 'laptop', 'headphones', 'smartwatch', 'tablet', 'camera', 'speaker',
            'keyboard', 'mouse', 'monitor', 'printer', 'router', 'hard drive']

POSITIVE_PHRASES = [
    "I love this {product}! It's been working great for months.",
    "This {product} exceeds my expectations. Highly recommend!",
    "Best {product} I've ever owned. The quality is outstanding.",
    "Fantastic {product}, very happy with my purchase.",
    "Great value for money. This {product} is perfect for my needs.",
    "This {product} is a game-changer! So glad I bought it.",
    "Excellent performance from this {product}. 5 stars!",
    "Very satisfied with this {product}. Would buy again.",
    "The {product} is incredibly well-designed and reliable.",
    "Amazing battery life on this {product}. Works like a charm!"
]

NEGATIVE_PHRASES = [
    "Disappointed with this {product}. Broke after just a few uses.",
    "This {product} is not worth the money. Save yourself the trouble.",
    "Terrible quality {product}. Would not recommend to anyone.",
    "The {product} stopped working after a week. Very frustrating!",
    "Poor customer service regarding this {product}. Will not buy again.",
    "This {product} is way overpriced for what you get.",
    "The {product} has constant issues that make it unusable.",
    "Worst {product} I've ever purchased. Complete waste of money.",
    "This {product} doesn't perform as advertised. Very misleading!",
    "Returned the {product} immediately. Absolutely horrible experience."
]

NEUTRAL_PHRASES = [
    "The {product} is okay. Nothing special but gets the job done.",
    "Average {product} for the price point. Not amazing, not terrible.",
    "This {product} has both pros and cons. Works well enough.",
    "The {product} meets basic expectations but doesn't wow me.",
    "Decent {product} overall. Some features could be improved.",
    "The {product} performs as expected. No surprises either way.",
    "Got this {product} on sale. It's acceptable for what I paid.",
    "Middle-of-the-road {product}. Neither disappointing nor impressive.",
    "This {product} is functional but lacks some features I was hoping for.",
    "The {product} is fine. Not sure I would buy it again though."
]


def generate_sample_reviews(n=200):
    reviews = []

    # Distribution of sentiments (slightly more positive than negative)
    sentiment_weights = [0.55, 0.25, 0.2]  # positive, neutral, negative

    for _ in range(n):
        product = random.choice(PRODUCTS)
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=sentiment_weights)

        if sentiment == 'positive':
            review_text = random.choice(POSITIVE_PHRASES).format(product=product)
        elif sentiment == 'negative':
            review_text = random.choice(NEGATIVE_PHRASES).format(product=product)
        else:
            review_text = random.choice(NEUTRAL_PHRASES).format(product=product)

        # Add some randomness to review length and variation
        if random.random() < 0.3:  # 30% chance of adding details
            extra_details = [
                f" The design is {'great' if sentiment == 'positive' else 'poor' if sentiment == 'negative' else 'okay'}.",
                f" {'Love' if sentiment == 'positive' else 'Hate' if sentiment == 'negative' else 'Neutral about'} the features.",
                f" Would{'' if sentiment == 'positive' else ' not' if sentiment == 'negative' else ' maybe'} buy again."
            ]
            review_text += random.choice(extra_details)

        # Add ratings
        if sentiment == 'positive':
            rating = random.randint(4, 5)
        elif sentiment == 'negative':
            rating = random.randint(1, 2)
        else:
            rating = 3

        reviews.append({
            'product_type': product,
            'review_text': review_text,
            'rating': rating,
            'actual_sentiment': sentiment  # For validation purposes
        })

    return pd.DataFrame(reviews)


def main():
    # Generate sample data
    df = generate_sample_reviews(200)

    # Save to CSV
    df.to_csv('sample_product_reviews.csv', index=False)
    print("Generated 200 sample product reviews and saved to 'sample_product_reviews.csv'")
    print(f"Distribution: {df['actual_sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()