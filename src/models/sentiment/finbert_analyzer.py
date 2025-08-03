"""
FinBERT-based sentiment analysis for financial text
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer
)
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class FinBERTAnalyzer:
    """
    Financial sentiment analysis using FinBERT
    """
    
    def __init__(
        self,
        model_name: str = 'ProsusAI/finbert',
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        Initialize FinBERT analyzer
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self._load_model()
        
    def _load_model(self) -> None:
        """Load FinBERT model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easy inference
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                max_length=self.max_length,
                truncation=True
            )
            
            # Get label mappings
            self.label_map = {
                0: 'negative',
                1: 'neutral',
                2: 'positive'
            }
            
            self.label_to_id = {v: k for k, v in self.label_map.items()}
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def analyze_sentiment(
        self,
        texts: Union[str, List[str]],
        return_all_scores: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Analyze sentiment of financial text
        
        Args:
            texts: Single text or list of texts
            return_all_scores: Return scores for all classes
            
        Returns:
            Sentiment analysis results
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Get predictions
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Process each result
                for j, (pred, prob) in enumerate(zip(predictions, probs)):
                    sentiment_label = self.label_map[pred.item()]
                    confidence = prob[pred].item()
                    
                    result = {
                        'text': batch[j][:100] + '...' if len(batch[j]) > 100 else batch[j],
                        'sentiment': sentiment_label,
                        'confidence': confidence,
                        'sentiment_score': self._calculate_sentiment_score(prob)
                    }
                    
                    if return_all_scores:
                        result['scores'] = {
                            'negative': prob[0].item(),
                            'neutral': prob[1].item(),
                            'positive': prob[2].item()
                        }
                    
                    results.append(result)
        
        return results[0] if single_input else results
    
    def _calculate_sentiment_score(self, probs: torch.Tensor) -> float:
        """
        Calculate normalized sentiment score (-1 to 1)
        
        Args:
            probs: Probability tensor
            
        Returns:
            Sentiment score
        """
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        
        # Calculate weighted score
        # negative: -1, neutral: 0, positive: 1
        weights = np.array([-1, 0, 1])
        score = np.dot(probs_np, weights)
        
        return float(score)
    
    def analyze_news_sentiment(
        self,
        news_df: pd.DataFrame,
        text_column: str = 'content',
        title_column: Optional[str] = 'title'
    ) -> pd.DataFrame:
        """
        Analyze sentiment of news articles
        
        Args:
            news_df: DataFrame with news articles
            text_column: Column containing article text
            title_column: Column containing article title
            
        Returns:
            DataFrame with sentiment scores
        """
        df = news_df.copy()
        
        # Combine title and content if both available
        if title_column and title_column in df.columns:
            texts = (df[title_column] + '. ' + df[text_column]).tolist()
        else:
            texts = df[text_column].tolist()
        
        # Analyze sentiment
        logger.info(f"Analyzing sentiment for {len(texts)} articles")
        
        results = self.analyze_sentiment(texts, return_all_scores=True)
        
        # Add results to dataframe
        df['sentiment'] = [r['sentiment'] for r in results]
        df['sentiment_confidence'] = [r['confidence'] for r in results]
        df['sentiment_score'] = [r['sentiment_score'] for r in results]
        
        # Add individual class scores
        df['score_negative'] = [r['scores']['negative'] for r in results]
        df['score_neutral'] = [r['scores']['neutral'] for r in results]
        df['score_positive'] = [r['scores']['positive'] for r in results]
        
        return df
    
    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        output_dir: str = './finbert_finetuned',
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Fine-tune FinBERT on custom data
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            
        Returns:
            Training results
        """
        logger.info("Preparing data for fine-tuning")
        
        # Convert labels to ids
        train_label_ids = [self.label_to_id[label] for label in train_labels]
        
        # Tokenize texts
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        
        # Create dataset
        train_dataset = FinBERTDataset(train_encodings, train_label_ids)
        
        # Prepare validation dataset if provided
        val_dataset = None
        if val_texts and val_labels:
            val_label_ids = [self.label_to_id[label] for label in val_labels]
            val_encodings = self.tokenizer(
                val_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            val_dataset = FinBERTDataset(val_encodings, val_label_ids)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model='f1' if val_dataset else None
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics if val_dataset else None
        )
        
        # Train
        logger.info("Starting fine-tuning")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
        # Evaluate on validation set
        results = {}
        if val_dataset:
            eval_results = trainer.evaluate()
            results['validation'] = eval_results
        
        return results
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def get_market_sentiment(
        self,
        news_df: pd.DataFrame,
        symbol_column: str = 'symbols',
        time_column: str = 'published_date',
        window: str = '24H'
    ) -> pd.DataFrame:
        """
        Calculate aggregated market sentiment by symbol
        
        Args:
            news_df: DataFrame with analyzed news
            symbol_column: Column containing symbols
            time_column: Column containing timestamps
            window: Time window for aggregation
            
        Returns:
            DataFrame with market sentiment by symbol
        """
        # Ensure sentiment scores exist
        if 'sentiment_score' not in news_df.columns:
            news_df = self.analyze_news_sentiment(news_df)
        
        # Explode symbols if they're lists
        if news_df[symbol_column].apply(lambda x: isinstance(x, list)).any():
            news_df = news_df.explode(symbol_column)
        
        # Set time index
        news_df[time_column] = pd.to_datetime(news_df[time_column])
        
        # Group by symbol and calculate metrics
        sentiment_summary = []
        
        for symbol in news_df[symbol_column].unique():
            symbol_news = news_df[news_df[symbol_column] == symbol]
            
            # Calculate aggregate metrics
            summary = {
                'symbol': symbol,
                'article_count': len(symbol_news),
                'avg_sentiment_score': symbol_news['sentiment_score'].mean(),
                'sentiment_std': symbol_news['sentiment_score'].std(),
                'positive_ratio': (symbol_news['sentiment'] == 'positive').mean(),
                'negative_ratio': (symbol_news['sentiment'] == 'negative').mean(),
                'neutral_ratio': (symbol_news['sentiment'] == 'neutral').mean(),
                'avg_confidence': symbol_news['sentiment_confidence'].mean(),
                'latest_sentiment': symbol_news.iloc[0]['sentiment_score'] if len(symbol_news) > 0 else 0
            }
            
            # Calculate momentum (change in sentiment)
            if len(symbol_news) > 1:
                recent = symbol_news.iloc[:len(symbol_news)//2]['sentiment_score'].mean()
                older = symbol_news.iloc[len(symbol_news)//2:]['sentiment_score'].mean()
                summary['sentiment_momentum'] = recent - older
            else:
                summary['sentiment_momentum'] = 0
            
            sentiment_summary.append(summary)
        
        return pd.DataFrame(sentiment_summary)
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save fine-tuned model"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': self.device,
            'label_map': self.label_map
        }
        
        with open(path / 'finbert_config.json', 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """Load fine-tuned model"""
        path = Path(path)
        
        # Load configuration
        with open(path / 'finbert_config.json', 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.max_length = config['max_length']
        self.label_map = config['label_map']
        self.label_to_id = {v: k for k, v in self.label_map.items()}
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        
        # Recreate pipeline
        self.sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1,
            max_length=self.max_length,
            truncation=True
        )
        
        logger.info(f"Model loaded from {path}")


class FinBERTDataset(torch.utils.data.Dataset):
    """Custom dataset for FinBERT fine-tuning"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)