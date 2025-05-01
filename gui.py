# gui.py
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import pandas as pd
import threading
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sentiment_analyzer import SentimentAnalyzer


class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Review Sentiment Analysis")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Set up the analyzer
        self.analyzer = SentimentAnalyzer()

        # Create and configure the notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.single_review_tab = ttk.Frame(self.notebook)
        self.batch_analysis_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.single_review_tab, text="Single Review Analysis")
        self.notebook.add(self.batch_analysis_tab, text="Batch Analysis")
        self.notebook.add(self.results_tab, text="Results & Visualizations")

        # Set up the single review analysis tab
        self.setup_single_review_tab()

        # Set up the batch analysis tab
        self.setup_batch_analysis_tab()

        # Set up the results tab
        self.setup_results_tab()

        # Initialize results storage
        self.batch_results_df = None

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_single_review_tab(self):
        # Review input area
        input_frame = ttk.LabelFrame(self.single_review_tab, text="Enter Product Review")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.review_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=8)
        self.review_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Analysis button
        button_frame = ttk.Frame(self.single_review_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.analyze_button = ttk.Button(button_frame, text="Analyze Sentiment", command=self.analyze_single_review)
        self.analyze_button.pack(side=tk.RIGHT)

        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_review_text)
        self.clear_button.pack(side=tk.RIGHT, padx=5)

        # Results area
        results_frame = ttk.LabelFrame(self.single_review_tab, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Overall sentiment display
        sentiment_frame = ttk.Frame(results_frame)
        sentiment_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sentiment_frame, text="Overall Sentiment:").pack(side=tk.LEFT)
        self.sentiment_var = tk.StringVar()
        self.sentiment_label = ttk.Label(sentiment_frame, textvariable=self.sentiment_var, font=("", 12, "bold"))
        self.sentiment_label.pack(side=tk.LEFT, padx=5)

        # Scores display
        scores_frame = ttk.Frame(results_frame)
        scores_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create score bars
        self.setup_score_bars(scores_frame)

    def setup_score_bars(self, parent):
        # Compound score
        ttk.Label(parent, text="Compound:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.compound_var = tk.DoubleVar()
        self.compound_bar = ttk.Progressbar(parent, variable=self.compound_var, length=300, mode='determinate')
        self.compound_bar.grid(row=0, column=1, padx=5, pady=2)
        self.compound_val = ttk.Label(parent, text="0.0")
        self.compound_val.grid(row=0, column=2, padx=5)

        # Positive score
        ttk.Label(parent, text="Positive:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.positive_var = tk.DoubleVar()
        self.positive_bar = ttk.Progressbar(parent, variable=self.positive_var, length=300, mode='determinate')
        self.positive_bar.grid(row=1, column=1, padx=5, pady=2)
        self.positive_val = ttk.Label(parent, text="0.0")
        self.positive_val.grid(row=1, column=2, padx=5)

        # Neutral score
        ttk.Label(parent, text="Neutral:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.neutral_var = tk.DoubleVar()
        self.neutral_bar = ttk.Progressbar(parent, variable=self.neutral_var, length=300, mode='determinate')
        self.neutral_bar.grid(row=2, column=1, padx=5, pady=2)
        self.neutral_val = ttk.Label(parent, text="0.0")
        self.neutral_val.grid(row=2, column=2, padx=5)

        # Negative score
        ttk.Label(parent, text="Negative:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.negative_var = tk.DoubleVar()
        self.negative_bar = ttk.Progressbar(parent, variable=self.negative_var, length=300, mode='determinate')
        self.negative_bar.grid(row=3, column=1, padx=5, pady=2)
        self.negative_val = ttk.Label(parent, text="0.0")
        self.negative_val.grid(row=3, column=2, padx=5)

    def setup_batch_analysis_tab(self):
        # File selection area
        file_frame = ttk.Frame(self.batch_analysis_tab)
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(file_frame, text="CSV File:").pack(side=tk.LEFT)

        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT, padx=5)

        # Options area
        options_frame = ttk.LabelFrame(self.batch_analysis_tab, text="Analysis Options")
        options_frame.pack(fill=tk.X, padx=10, pady=10)

        # Column selection
        col_frame = ttk.Frame(options_frame)
        col_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(col_frame, text="Review Text Column:").pack(side=tk.LEFT)
        self.column_var = tk.StringVar(value="Auto-detect")
        self.column_combo = ttk.Combobox(col_frame, textvariable=self.column_var, state="readonly")
        self.column_combo.pack(side=tk.LEFT, padx=5)
        self.column_combo['values'] = ["Auto-detect"]

        # Output file option
        out_frame = ttk.Frame(options_frame)
        out_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(out_frame, text="Output File:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value="analysis_results.csv")
        self.output_entry = ttk.Entry(out_frame, textvariable=self.output_var, width=30)
        self.output_entry.pack(side=tk.LEFT, padx=5)

        # Analyze button
        button_frame = ttk.Frame(self.batch_analysis_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.batch_analyze_button = ttk.Button(button_frame, text="Analyze Reviews", command=self.analyze_batch)
        self.batch_analyze_button.pack(side=tk.RIGHT)

        # Preview area
        preview_frame = ttk.LabelFrame(self.batch_analysis_tab, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a treeview for data preview
        self.preview_tree = ttk.Treeview(preview_frame)
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)

        # Progress bar for batch processing
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.batch_analysis_tab, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)


    def setup_results_tab(self):
        """Set up the results tab with visualizations and metrics"""
        # Create a frame for the visualizations
        viz_frame = ttk.Frame(self.results_tab)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a frame for the plots
        self.plot_frame = ttk.Frame(viz_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create a frame for the summary
        summary_frame = ttk.LabelFrame(self.results_tab, text="Analysis Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)

        # Summary labels
        self.summary_total = ttk.Label(summary_frame, text="Total Reviews: 0")
        self.summary_total.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

        self.summary_positive = ttk.Label(summary_frame, text="Positive: 0 (0.0%)")
        self.summary_positive.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        self.summary_neutral = ttk.Label(summary_frame, text="Neutral: 0 (0.0%)")
        self.summary_neutral.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)

        self.summary_negative = ttk.Label(summary_frame, text="Negative: 0 (0.0%)")
        self.summary_negative.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        # Create a frame for metrics
        metrics_frame = ttk.LabelFrame(self.results_tab, text="Model Performance Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)

        # Metrics labels - will be populated when available
        self.metrics_available = ttk.Label(metrics_frame, text="No ground truth data available for metrics calculation")
        self.metrics_available.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)

        self.accuracy_label = ttk.Label(metrics_frame, text="Accuracy: N/A")
        self.accuracy_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)

        self.precision_label = ttk.Label(metrics_frame, text="Precision: N/A")
        self.precision_label.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        self.f1_label = ttk.Label(metrics_frame, text="F1 Score: N/A")
        self.f1_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)

        # Buttons for exporting/saving
        button_frame = ttk.Frame(self.results_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.save_plot_button = ttk.Button(button_frame, text="Save Visualizations", command=self.save_visualizations)
        self.save_plot_button.pack(side=tk.RIGHT, padx=5)

        self.export_button = ttk.Button(button_frame, text="Export Results", command=self.export_results)
        self.export_button.pack(side=tk.RIGHT, padx=5)

    def clear_review_text(self):
        """Clear the review text area"""
        self.review_text.delete(1.0, tk.END)

    def browse_file(self):
        """Open file dialog to select a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.load_csv_preview(file_path)

    def load_csv_preview(self, file_path):
        """Load and display a preview of the CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Update column dropdown
            columns = list(df.columns)
            self.column_combo['values'] = ["Auto-detect"] + columns

            # Clear existing tree items
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)

            # Configure tree columns
            self.preview_tree['columns'] = columns
            self.preview_tree['show'] = 'headings'

            for col in columns:
                self.preview_tree.heading(col, text=col)
                # Adjust column width based on content
                max_width = max(len(str(col)), df[col].astype(str).str.len().max())
                self.preview_tree.column(col, width=min(max_width * 8, 300))

            # Add data rows (first 5 rows for preview)
            for i, row in df.head(5).iterrows():
                values = [str(row[col]) for col in columns]
                self.preview_tree.insert('', 'end', values=values)

            self.status_var.set(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
            self.status_var.set("Error loading file")

    def analyze_single_review(self):
        """Analyze a single review"""
        review_text = self.review_text.get(1.0, tk.END).strip()

        if not review_text:
            messagebox.showinfo("Info", "Please enter a review to analyze.")
            return

        # Perform sentiment analysis
        results = self.analyzer.analyze_text(review_text)

        # Update the UI with results
        sentiment = results['sentiment']
        self.sentiment_var.set(sentiment.capitalize())

        # Update the sentiment label color
        if sentiment == 'positive':
            self.sentiment_label.configure(foreground='green')
        elif sentiment == 'negative':
            self.sentiment_label.configure(foreground='red')
        else:
            self.sentiment_label.configure(foreground='gray')

        # Update score bars
        self.update_score_bars(results)

        self.status_var.set("Analysis completed")

    def update_score_bars(self, results):
        """Update the score progress bars"""
        # Update compound score (-1 to 1, scaled to 0-100 for progress bar)
        compound = results['compound']
        self.compound_var.set((compound + 1) * 50)  # Scale from -1,1 to 0,100
        self.compound_val.configure(text=f"{compound:.3f}")

        # Update positive score
        positive = results['pos']
        self.positive_var.set(positive * 100)
        self.positive_val.configure(text=f"{positive:.3f}")

        # Update neutral score
        neutral = results['neu']
        self.neutral_var.set(neutral * 100)
        self.neutral_val.configure(text=f"{neutral:.3f}")

        # Update negative score
        negative = results['neg']
        self.negative_var.set(negative * 100)
        self.negative_val.configure(text=f"{negative:.3f}")

    def analyze_batch(self):
        """Analyze reviews in batch mode"""
        file_path = self.file_path_var.get()

        if not file_path:
            messagebox.showinfo("Info", "Please select a CSV file.")
            return

        try:
            # Disable UI elements during processing
            self.batch_analyze_button.configure(state='disabled')
            self.status_var.set("Loading data...")
            self.progress_var.set(0)

            # Start analysis in a separate thread to keep UI responsive
            threading.Thread(target=self._run_batch_analysis, args=(file_path,), daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")
            self.batch_analyze_button.configure(state='normal')
            self.status_var.set("Error starting analysis")

    def _run_batch_analysis(self, file_path):
        """Run the batch analysis in a separate thread"""
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            self.root.after(0, lambda: self.status_var.set(f"Loaded {len(df)} rows. Determining text column..."))

            # Determine which column to use for reviews
            column = self.column_var.get()
            if column == "Auto-detect":
                # Try to find a suitable text column
                possible_columns = ['review', 'text', 'content', 'review_text', 'comment']
                found = False
                for col in possible_columns:
                    if col in df.columns:
                        column = col
                        found = True
                        break

                if not found and len(df.columns) > 0:
                    # Use the first column that looks like text
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            column = col
                            break

                if not found:
                    self.root.after(0, lambda: messagebox.showerror("Error",
                                                                    "Could not automatically detect a text column"))
                    self.root.after(0, lambda: self.batch_analyze_button.configure(state='normal'))
                    return

            self.root.after(0, lambda: self.status_var.set(f"Using column '{column}' for analysis..."))

            # Get the reviews
            reviews = df[column].fillna('').tolist()
            total = len(reviews)

            # Set up progress tracking
            self.root.after(0, lambda: self.progress_var.set(0))

            # Process in batches to update progress
            batch_size = max(1, min(100, total // 10))
            results = []

            for i in range(0, total, batch_size):
                # Process a batch
                end = min(i + batch_size, total)
                batch = reviews[i:end]

                # Analyze the batch
                for review in batch:
                    scores = self.analyzer.analyze_text(review)
                    results.append({
                        'review': review,
                        'sentiment': scores['sentiment'],
                        'compound_score': scores['compound'],
                        'positive_score': scores['pos'],
                        'neutral_score': scores['neu'],
                        'negative_score': scores['neg']
                    })

                # Update progress
                progress = min(100, int((end / total) * 100))
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda e=end, t=total: self.status_var.set(f"Processed {e}/{t} reviews..."))

            # Create results dataframe
            results_df = pd.DataFrame(results)

            # If the original df had additional columns, merge them
            if len(df.columns) > 1:
                # Reset index to allow merging
                df_reset = df.reset_index(drop=True)
                results_df = pd.concat([df_reset, results_df.drop('review', axis=1)], axis=1)

            self.batch_results_df = results_df

            # Save results to CSV
            output_file = self.output_var.get()
            results_df.to_csv(output_file, index=False)

            # Generate visualizations
            self.root.after(0, lambda: self.status_var.set("Generating visualizations..."))
            self.root.after(0, self.update_visualizations)

            # Switch to results tab
            self.root.after(0, lambda: self.notebook.select(self.results_tab))
            self.root.after(0, lambda: self.status_var.set(f"Analysis completed. Results saved to {output_file}"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error during analysis: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error during analysis"))

        finally:
            self.root.after(0, lambda: self.batch_analyze_button.configure(state='normal'))

    def update_visualizations(self):
        """Update the visualizations based on batch analysis results"""
        if self.batch_results_df is None or len(self.batch_results_df) == 0:
            return

        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Generate visualizations and get summary stats
        summary = self.analyzer.visualize_results(self.batch_results_df)

        # Create a figure with subplots
        if summary.get('has_metrics', False):
            # If we have metrics, create three plots (including confusion matrix)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

            # Plot sentiment distribution
            sentiment_counts = self.batch_results_df['sentiment'].value_counts()
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax1)
            ax1.set_title('Sentiment Distribution')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Sentiment')

            # Plot compound score distribution
            sns.histplot(self.batch_results_df['compound_score'], kde=True, ax=ax2)
            ax2.set_title('Compound Score Distribution')
            ax2.set_xlabel('Compound Score')
            ax2.set_ylabel('Frequency')

            # Plot confusion matrix
            sns.heatmap(summary['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['positive', 'neutral', 'negative'],
                        yticklabels=['positive', 'neutral', 'negative'],
                        ax=ax3)
            ax3.set_title('Confusion Matrix')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        else:
            # If no metrics, just show two plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Plot sentiment distribution
            sentiment_counts = self.batch_results_df['sentiment'].value_counts()
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax1)
            ax1.set_title('Sentiment Distribution')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Sentiment')

            # Plot compound score distribution
            sns.histplot(self.batch_results_df['compound_score'], kde=True, ax=ax2)
            ax2.set_title('Compound Score Distribution')
            ax2.set_xlabel('Compound Score')
            ax2.set_ylabel('Frequency')

        # Adjust layout
        plt.tight_layout()

        # Embed plots in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update summary statistics
        total = summary['total_reviews']
        pos_count = summary['positive']
        neu_count = summary['neutral']
        neg_count = summary['negative']

        pos_pct = summary['positive_pct']
        neu_pct = summary['neutral_pct']
        neg_pct = summary['negative_pct']

        self.summary_total.configure(text=f"Total Reviews: {total}")
        self.summary_positive.configure(text=f"Positive: {pos_count} ({pos_pct:.1f}%)")
        self.summary_neutral.configure(text=f"Neutral: {neu_count} ({neu_pct:.1f}%)")
        self.summary_negative.configure(text=f"Negative: {neg_count} ({neg_pct:.1f}%)")

        # Update metrics if available
        if summary.get('has_metrics', False):
            self.metrics_available.configure(text="Model Performance Metrics (Based on Ground Truth Data):")
            self.accuracy_label.configure(text=f"Accuracy: {summary['accuracy']:.3f}")
            self.precision_label.configure(text=f"Precision: {summary['precision']:.3f}")
            self.f1_label.configure(text=f"F1 Score: {summary['f1_score']:.3f}")
        else:
            self.metrics_available.configure(text="No ground truth data available for metrics calculation")
            self.accuracy_label.configure(text="Accuracy: N/A")
            self.precision_label.configure(text="Precision: N/A")
            self.f1_label.configure(text="F1 Score: N/A")

    def save_visualizations(self):
        """Save the visualizations to image files"""
        if self.batch_results_df is None:
            messagebox.showinfo("Info", "No analysis results available to save.")
            return

        try:
            # Generate summary and get metrics
            summary = self.analyzer.visualize_results(self.batch_results_df)

            # Create the visualizations in new figures
            plt.figure(figsize=(8, 6))
            sentiment_counts = self.batch_results_df['sentiment'].value_counts()
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
            plt.title('Sentiment Distribution')
            plt.ylabel('Count')
            plt.xlabel('Sentiment')
            plt.tight_layout()
            plt.savefig('sentiment_distribution.png')

            plt.figure(figsize=(8, 6))
            sns.histplot(self.batch_results_df['compound_score'], kde=True)
            plt.title('Compound Score Distribution')
            plt.xlabel('Compound Score')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig('compound_score_distribution.png')

            saved_files = ['sentiment_distribution.png', 'compound_score_distribution.png']

            # If metrics are available, save the confusion matrix
            if summary.get('has_metrics', False):
                plt.figure(figsize=(8, 6))
                sns.heatmap(summary['confusion_matrix'],
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            xticklabels=['positive', 'neutral', 'negative'],
                            yticklabels=['positive', 'neutral', 'negative'])
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Sentiment')
                plt.ylabel('Actual Sentiment')
                plt.tight_layout()
                plt.savefig('confusion_matrix.png')
                saved_files.append('confusion_matrix.png')

            messagebox.showinfo("Success", f"Visualizations saved as: {', '.join(saved_files)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save visualizations: {str(e)}")

    def export_results(self):
        """Export analysis results to a custom location"""
        if self.batch_results_df is None:
            messagebox.showinfo("Info", "No analysis results available to export.")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Results As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )

            if not file_path:
                return

            # Export based on file extension
            if file_path.endswith('.xlsx'):
                self.batch_results_df.to_excel(file_path, index=False)
            else:
                self.batch_results_df.to_csv(file_path, index=False)

            messagebox.showinfo("Success", f"Results exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")


def main():
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()