# main.py
import pandas as pd
import argparse
from sentiment_analyzer import SentimentAnalyzer


def load_reviews_from_csv(file_path):
    """Load reviews from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Check if there's a column that could contain reviews
        possible_review_columns = ['review', 'text', 'content', 'review_text', 'comment']
        for col in possible_review_columns:
            if col in df.columns:
                return df[col].fillna('').tolist()

        # If no specific column is found, use the first text column
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            return df[text_columns[0]].fillna('').tolist()

        raise ValueError("No suitable text column found in the CSV file")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []


# Updated portion of the main.py file

def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment in product reviews')
    parser.add_argument('--file', type=str, help='Path to CSV file containing reviews')
    parser.add_argument('--text', type=str, help='Single review text to analyze')
    parser.add_argument('--output', type=str, default='analysis_results.csv',
                        help='Output file for analysis results')

    args = parser.parse_args()
    analyzer = SentimentAnalyzer()

    if args.text:
        # Analyze a single review
        result = analyzer.analyze_text(args.text)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Scores: Positive={result['pos']:.3f}, Neutral={result['neu']:.3f}, Negative={result['neg']:.3f}")
        print(f"Compound Score: {result['compound']:.3f}")

    elif args.file:
        # Analyze reviews from a file
        reviews = load_reviews_from_csv(args.file)
        if reviews:
            print(f"Analyzing {len(reviews)} reviews...")
            results_df = analyzer.analyze_reviews(reviews)

            # Save results to CSV
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")

            # Generate visualizations and get summary
            summary = analyzer.visualize_results(results_df)
            print("\nSummary of Analysis:")
            print(f"Total Reviews: {summary['total_reviews']}")
            print(f"Positive: {summary['positive']} ({summary['positive_pct']:.1f}%)")
            print(f"Neutral: {summary['neutral']} ({summary['neutral_pct']:.1f}%)")
            print(f"Negative: {summary['negative']} ({summary['negative_pct']:.1f}%)")

            # Display metrics if available
            if summary.get('has_metrics', False):
                print("\nModel Performance Metrics:")
                print(f"Accuracy: {summary['accuracy']:.3f}")
                print(f"Precision: {summary['precision']:.3f}")
                print(f"F1 Score: {summary['f1_score']:.3f}")
                print("\nConfusion matrix saved to 'confusion_matrix.png'")

            print("\nVisualization saved as 'sentiment_distribution.png' and 'compound_score_distribution.png'")

    else:
        # Interactive mode
        print("Sentiment Analysis for Product Reviews")
        print("======================================")
        print("Enter 'quit' to exit")

        while True:
            review = input("\nEnter a review to analyze: ")
            if review.lower() == 'quit':
                break

            result = analyzer.analyze_text(review)
            print(f"Sentiment: {result['sentiment']}")
            print(f"Scores: Positive={result['pos']:.3f}, Neutral={result['neu']:.3f}, Negative={result['neg']:.3f}")
            print(f"Compound Score: {result['compound']:.3f}")

if __name__ == "__main__":
    main()