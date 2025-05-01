# sentiment_analyzer.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix


class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        """Analyze the sentiment of a single text."""
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0,
                'sentiment': 'neutral'
            }

        sentiment_scores = self.sia.polarity_scores(text)

        # Determine sentiment category
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        sentiment_scores['sentiment'] = sentiment
        return sentiment_scores

    def analyze_reviews(self, reviews):
        """Analyze multiple reviews and return results as a DataFrame."""
        results = []

        for review in reviews:
            scores = self.analyze_text(review)
            results.append({
                'review': review,
                'sentiment': scores['sentiment'],
                'compound_score': scores['compound'],
                'positive_score': scores['pos'],
                'neutral_score': scores['neu'],
                'negative_score': scores['neg']
            })

        return pd.DataFrame(results)

    def calculate_metrics(self, df):
        """Calculate classification metrics if actual sentiment is available."""
        metrics = {'has_metrics': False}

        # Check if the DataFrame has the actual_sentiment column
        if 'actual_sentiment' in df.columns:
            y_true = df['actual_sentiment']
            y_pred = df['sentiment']

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)

            # For precision and f1, we need to handle multiclass
            # Using weighted average for precision and f1 score
            precision = precision_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['positive', 'neutral', 'negative'])

            metrics = {
                'has_metrics': True,
                'accuracy': accuracy,
                'precision': precision,
                'f1_score': f1,
                'confusion_matrix': cm
            }

        return metrics

    def visualize_results(self, df):
        """Create visualizations for the sentiment analysis results."""
        # Count plot of sentiment categories
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment', data=df, palette={'positive': 'green', 'neutral': 'gray', 'negative': 'red'})
        plt.title('Distribution of Sentiment in Reviews')
        plt.savefig('sentiment_distribution.png')
        plt.close()

        # Distribution of compound scores
        plt.figure(figsize=(10, 6))
        sns.histplot(df['compound_score'], kde=True)
        plt.title('Distribution of Compound Sentiment Scores')
        plt.savefig('compound_score_distribution.png')
        plt.close()

        # Create a summary
        summary = df['sentiment'].value_counts().to_dict()
        positive_pct = (summary.get('positive', 0) / len(df)) * 100
        negative_pct = (summary.get('negative', 0) / len(df)) * 100
        neutral_pct = (summary.get('neutral', 0) / len(df)) * 100

        # Calculate metrics if actual_sentiment is available
        metrics = self.calculate_metrics(df)

        summary_data = {
            'total_reviews': len(df),
            'positive': summary.get('positive', 0),
            'negative': summary.get('negative', 0),
            'neutral': summary.get('neutral', 0),
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct
        }

        # Add metrics to summary if available
        if metrics['has_metrics']:
            summary_data.update({
                'has_metrics': True,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'f1_score': metrics['f1_score'],
                'confusion_matrix': metrics['confusion_matrix']
            })

            # Create confusion matrix plot if metrics are available
            plt.figure(figsize=(8, 6))
            sns.heatmap(metrics['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['positive', 'neutral', 'negative'],
                        yticklabels=['positive', 'neutral', 'negative'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Sentiment')
            plt.ylabel('Actual Sentiment')
            plt.savefig('confusion_matrix.png')
            plt.close()
        else:
            summary_data['has_metrics'] = False

        return summary_data