from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import cleantext
import re
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Configuration & Constants
# -----------------------
class Config:
    """Configuration settings for the analyzer."""
    MODELS = {
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "emotion": "j-hartmann/emotion-english-distilroberta-base",
        "toxicity": "martin-ha/toxic-comment-model",
        "intent": "joeddav/xlm-roberta-large-xnli"  # For intent classification
    }
    
    CACHE_SIZE = 1000
    MAX_TEXT_LENGTH = 512
    BATCH_SIZE = 8
    CONFIDENCE_THRESHOLD = 0.1
    
    # Emotion mappings for better categorization
    EMOTION_CATEGORIES = {
        "positive": ["joy", "surprise", "love"],
        "negative": ["anger", "fear", "sadness", "disgust"],
        "neutral": ["neutral"]
    }

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emosenai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EmoSenAI")

# -----------------------
# 1. Advanced Text Cleaning & Preprocessing
# -----------------------
class TextPreprocessor:
    """Advanced text preprocessing with multiple cleaning strategies."""
    
    @staticmethod
    def clean_text(text: str, aggressive: bool = False) -> str:
        """
        Advanced text cleaning with multiple levels.
        
        Args:
            text: Input text to clean
            aggressive: Whether to use aggressive cleaning (removes more content)
        """
        if not text or not isinstance(text, str):
            return ""
            
        original_length = len(text)
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove URLs, emails, phone numbers
        text = re.sub(r"http\S+", "[URL]", text)
        text = re.sub(r"@\w+", "[MENTION]", text)
        text = re.sub(r"#\w+", "[HASHTAG]", text)
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)
        text = re.sub(r"[\+]?[1-9][\d]{0,2}[\s]?[\(]?[\d]{3}[\)]?[\s]?[\d]{3}[\s]?[\d]{4}", "[PHONE]", text)
        
        if aggressive:
            # Aggressive cleaning - remove most special characters
            text = re.sub(r"[^\w\s.!?]", "", text)
        else:
            # Conservative cleaning - keep basic punctuation
            text = re.sub(r"[^\w\s.,!?']", "", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Use cleantext for additional normalization
        try:
            text = cleantext.clean(
                text,
                fix_unicode=True,
                to_ascii=True,
                lower=True,
                no_line_breaks=True,
                no_urls=False,  # Already handled
                no_emails=False,  # Already handled
                no_phone_numbers=False,  # Already handled
                no_numbers=False,
                no_digits=False,
                no_currency_symbols=True,
                no_punct=False,
                lang="en"
            )
        except Exception as e:
            logger.warning(f"Cleantext processing failed: {e}")
        
        cleaned_length = len(text)
        logger.debug(f"Text cleaned: {original_length} -> {cleaned_length} characters")
        
        return text.strip()
    
    @staticmethod
    def segment_text(text: str, max_length: int = Config.MAX_TEXT_LENGTH) -> List[str]:
        """Segment long text into chunks for processing."""
        words = text.split()
        segments = []
        current_segment = []
        
        for word in words:
            if len(' '.join(current_segment + [word])) <= max_length:
                current_segment.append(word)
            else:
                segments.append(' '.join(current_segment))
                current_segment = [word]
        
        if current_segment:
            segments.append(' '.join(current_segment))
            
        return segments
    
    @staticmethod
    def extract_features(text: str) -> Dict:
        """Extract basic text features."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0,
            "has_questions": any('?' in sentence for sentence in sentences),
            "has_exclamations": any('!' in sentence for sentence in sentences),
            "readability_score": TextPreprocessor.calculate_readability(text)
        }
    
    @staticmethod
    def calculate_readability(text: str) -> float:
        """Calculate simple readability score."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula
        return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)

# -----------------------
# 2. Advanced Model Manager with Caching & Fallbacks
# -----------------------
class ModelManager:
    """Manages multiple models with caching and fallback mechanisms."""
    
    def __init__(self):
        self.models = {}
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._load_models()
    
    def _load_models(self):
        """Load all models with error handling."""
        logger.info("Loading AI models...")
        
        models_to_load = {
            "sentiment": pipeline(
                "text-classification",
                model=Config.MODELS["sentiment"],
                return_all_scores=True,
                truncation=True,
                max_length=Config.MAX_TEXT_LENGTH
            ),
            "emotion": pipeline(
                "text-classification", 
                model=Config.MODELS["emotion"],
                return_all_scores=True,
                truncation=True,
                max_length=Config.MAX_TEXT_LENGTH
            ),
            "toxicity": pipeline(
                "text-classification",
                model=Config.MODELS["toxicity"],
                return_all_scores=True,
                truncation=True
            )
        }
        
        for name, model in models_to_load.items():
            try:
                self.models[name] = model
                logger.info(f"✅ {name.capitalize()} model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load {name} model: {e}")
                self.models[name] = None
        
        logger.info("All models loaded!")
    
    def _get_cache_key(self, text: str, model_type: str) -> str:
        """Generate cache key."""
        return f"{model_type}:{hash(text)}"
    
    def _check_cache(self, key: str) -> Optional[Dict]:
        """Check if result is in cache."""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def _update_cache(self, key: str, result: Dict):
        """Update cache with new result."""
        if len(self.cache) >= Config.CACHE_SIZE:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = result
    
    async def analyze_batch(self, texts: List[str], model_type: str) -> List[Dict]:
        """Analyze multiple texts asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: [self.analyze_text(text, model_type) for text in texts]
        )
    
    def analyze_text(self, text: str, model_type: str) -> Dict:
        """Analyze single text with caching."""
        if model_type not in self.models or self.models[model_type] is None:
            return {"error": f"Model {model_type} not available"}
        
        cache_key = self._get_cache_key(text, model_type)
        cached_result = self._check_cache(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            start_time = time.time()
            results = self.models[model_type](text)
            processing_time = time.time() - start_time
            
            if model_type in ["sentiment", "emotion", "toxicity"]:
                sorted_results = sorted(results[0], key=lambda x: x["score"], reverse=True)
                result = {
                    "predictions": sorted_results,
                    "top_prediction": sorted_results[0],
                    "processing_time": round(processing_time, 4),
                    "cached": False
                }
            else:
                result = results
            
            self._update_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {model_type}: {e}")
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }

# -----------------------
# 3. Advanced Analysis Engine
# -----------------------
class AdvancedTextAnalyzer:
    """Main analysis engine with multiple analysis types."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.preprocessor = TextPreprocessor()
        self.analysis_history = []
    
    def comprehensive_analysis(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis including:
        - Sentiment analysis
        - Emotion detection  
        - Toxicity detection
        - Text features
        - Combined insights
        """
        logger.info(f"Starting comprehensive analysis for text: {text[:100]}...")
        
        # Clean and preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        features = self.preprocessor.extract_features(cleaned_text)
        
        # Parallel analysis for different aspects
        sentiment_result = self.model_manager.analyze_text(cleaned_text, "sentiment")
        emotion_result = self.model_manager.analyze_text(cleaned_text, "emotion")
        toxicity_result = self.model_manager.analyze_text(cleaned_text, "toxicity")
        
        # Process results
        analysis_result = self._process_results(
            text, cleaned_text, sentiment_result, emotion_result, toxicity_result, features
        )
        
        # Store in history
        self.analysis_history.append(analysis_result)
        if len(self.analysis_history) > 1000:  # Limit history size
            self.analysis_history.pop(0)
        
        return analysis_result
    
    def _process_results(self, original_text: str, cleaned_text: str, 
                        sentiment_result: Dict, emotion_result: Dict, 
                        toxicity_result: Dict, features: Dict) -> Dict:
        """Process and combine all analysis results."""
        
        # Extract top predictions
        sentiment_top = sentiment_result.get("top_prediction", {})
        emotion_top = emotion_result.get("top_prediction", {})
        toxicity_top = toxicity_result.get("top_prediction", {})
        
        # Calculate combined metrics
        sentiment_label = sentiment_top.get("label", "neutral").upper()
        emotion_label = emotion_top.get("label", "neutral").capitalize()
        
        # Determine emotional category
        emotion_category = "neutral"
        for category, emotions in Config.EMOTION_CATEGORIES.items():
            if emotion_label.lower() in emotions:
                emotion_category = category
                break
        
        # Calculate confidence scores
        sentiment_confidence = sentiment_top.get("score", 0) * 100
        emotion_confidence = emotion_top.get("score", 0) * 100
        toxicity_confidence = toxicity_top.get("score", 0) * 100
        
        # Generate insights
        insights = self._generate_insights(
            sentiment_label, emotion_label, emotion_category, 
            sentiment_confidence, emotion_confidence, features
        )
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "original_length": len(original_text),
                "cleaned_length": len(cleaned_text),
                "processing_time": {
                    "sentiment": sentiment_result.get("processing_time", 0),
                    "emotion": emotion_result.get("processing_time", 0),
                    "toxicity": toxicity_result.get("processing_time", 0)
                }
            },
            "text": {
                "original": original_text,
                "cleaned": cleaned_text,
                "features": features
            },
            "analysis": {
                "sentiment": {
                    "predicted": sentiment_label,
                    "confidence": round(sentiment_confidence, 2),
                    "all_scores": {
                        item["label"]: round(item["score"], 4) 
                        for item in sentiment_result.get("predictions", [])
                    }
                },
                "emotion": {
                    "predicted": emotion_label,
                    "category": emotion_category,
                    "confidence": round(emotion_confidence, 2),
                    "all_scores": {
                        item["label"]: round(item["score"], 4) 
                        for item in emotion_result.get("predictions", [])
                    }
                },
                "toxicity": {
                    "predicted": toxicity_top.get("label", "neutral"),
                    "confidence": round(toxicity_confidence, 2),
                    "is_toxic": toxicity_confidence > 50,
                    "all_scores": {
                        item["label"]: round(item["score"], 4) 
                        for item in toxicity_result.get("predictions", [])
                    }
                }
            },
            "insights": insights,
            "combined_metrics": {
                "overall_sentiment_score": self._calculate_overall_score(sentiment_result, emotion_result),
                "emotional_intensity": max(emotion_confidence, sentiment_confidence) / 100,
                "risk_level": "high" if toxicity_confidence > 70 else "medium" if toxicity_confidence > 30 else "low"
            }
        }
    
    def _generate_insights(self, sentiment: str, emotion: str, emotion_category: str,
                          sentiment_conf: float, emotion_conf: float, features: Dict) -> List[str]:
        """Generate intelligent insights based on analysis results."""
        insights = []
        
        # Sentiment-based insights
        if sentiment_conf > 80:
            if "POSITIVE" in sentiment:
                insights.append("Strong positive sentiment detected - very favorable tone")
            elif "NEGATIVE" in sentiment:
                insights.append("Strong negative sentiment detected - concerning tone")
        
        # Emotion-based insights
        if emotion_conf > 75:
            if emotion.lower() in ["joy", "love"]:
                insights.append("High levels of positive emotion - indicates happiness or satisfaction")
            elif emotion.lower() in ["anger", "disgust"]:
                insights.append("Strong negative emotions detected - may indicate frustration or dissatisfaction")
        
        # Text feature insights
        if features["has_exclamations"]:
            insights.append("Contains exclamations - indicates emphasis or strong feeling")
        if features["has_questions"]:
            insights.append("Contains questions - seeking information or engagement")
        
        # Combined insights
        if "POSITIVE" in sentiment and emotion_category == "positive":
            insights.append("Consistent positive expression - genuine positive engagement")
        elif "NEGATIVE" in sentiment and emotion_category == "negative":
            insights.append("Consistent negative expression - genuine concern or dissatisfaction")
        elif "POSITIVE" in sentiment and emotion_category == "negative":
            insights.append("Mixed signals - positive words with negative emotions detected")
        
        # Readability insights
        readability = features["readability_score"]
        if readability > 70:
            insights.append("High readability - easy to understand")
        elif readability < 30:
            insights.append("Low readability - may be difficult to comprehend")
        
        return insights if insights else ["Text shows balanced characteristics"]
    
    def _calculate_overall_score(self, sentiment_result: Dict, emotion_result: Dict) -> float:
        """Calculate overall sentiment-emotion score."""
        sentiment_scores = sentiment_result.get("predictions", [])
        emotion_scores = emotion_result.get("predictions", [])
        
        if not sentiment_scores or not emotion_scores:
            return 0.5
        
        # Weighted combination of sentiment and emotion
        sentiment_weight = 0.6
        emotion_weight = 0.4
        
        # Convert to numerical scores (-1 to 1 scale)
        sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
        emotion_map = {"anger": -1, "fear": -0.5, "sadness": -0.8, "disgust": -1, 
                      "neutral": 0, "surprise": 0.3, "joy": 1, "love": 1}
        
        sentiment_score = 0
        for item in sentiment_scores:
            label = item["label"].lower()
            score = item["score"]
            sentiment_score += sentiment_map.get(label, 0) * score
        
        emotion_score = 0
        for item in emotion_scores:
            label = item["label"].lower()
            score = item["score"]
            emotion_score += emotion_map.get(label, 0) * score
        
        # Combine scores
        overall = (sentiment_score * sentiment_weight) + (emotion_score * emotion_weight)
        
        # Normalize to 0-1 scale
        return (overall + 1) / 2
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts in batch."""
        return [self.comprehensive_analysis(text) for text in texts]
    
    def get_analysis_stats(self) -> Dict:
        """Get statistics about analyses performed."""
        if not self.analysis_history:
            return {}
        
        sentiments = [item["analysis"]["sentiment"]["predicted"] for item in self.analysis_history]
        emotions = [item["analysis"]["emotion"]["predicted"] for item in self.analysis_history]
        
        return {
            "total_analyses": len(self.analysis_history),
            "most_common_sentiment": max(set(sentiments), key=sentiments.count),
            "most_common_emotion": max(set(emotions), key=emotions.count),
            "average_confidence": {
                "sentiment": np.mean([item["analysis"]["sentiment"]["confidence"] for item in self.analysis_history]),
                "emotion": np.mean([item["analysis"]["emotion"]["confidence"] for item in self.analysis_history])
            },
            "cache_stats": self.model_manager.get_cache_stats()
        }

# -----------------------
# 4. Export & Reporting
# -----------------------
class ReportGenerator:
    """Generate reports and exports from analysis results."""
    
    @staticmethod
    def to_dataframe(analysis_results: List[Dict]) -> pd.DataFrame:
        """Convert analysis results to pandas DataFrame."""
        rows = []
        for result in analysis_results:
            row = {
                "timestamp": result["metadata"]["timestamp"],
                "original_text": result["text"]["original"],
                "cleaned_text": result["text"]["cleaned"],
                "sentiment": result["analysis"]["sentiment"]["predicted"],
                "sentiment_confidence": result["analysis"]["sentiment"]["confidence"],
                "emotion": result["analysis"]["emotion"]["predicted"],
                "emotion_confidence": result["analysis"]["emotion"]["confidence"],
                "toxicity": result["analysis"]["toxicity"]["predicted"],
                "toxicity_confidence": result["analysis"]["toxicity"]["confidence"],
                "is_toxic": result["analysis"]["toxicity"]["is_toxic"],
                "word_count": result["text"]["features"]["word_count"],
                "readability_score": result["text"]["features"]["readability_score"],
                "risk_level": result["combined_metrics"]["risk_level"],
                "overall_score": result["combined_metrics"]["overall_sentiment_score"]
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def generate_report(analyzer: AdvancedTextAnalyzer, filename: str = None):
        """Generate a comprehensive report."""
        if not filename:
            filename = f"emosenai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis_statistics": analyzer.get_analysis_stats(),
            "recent_analyses": analyzer.analysis_history[-10:],  # Last 10 analyses
            "system_metrics": {
                "models_loaded": list(analyzer.model_manager.models.keys()),
                "cache_performance": analyzer.model_manager.get_cache_stats()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report generated: {filename}")
        return filename

# -----------------------
# 5. Advanced CLI Interface
# -----------------------
class AdvancedCLI:
    """Advanced command-line interface with multiple modes."""
    
    def __init__(self):
        self.analyzer = AdvancedTextAnalyzer()
        self.report_generator = ReportGenerator()
    
    def display_results(self, result: Dict):
        """Display analysis results in a formatted way."""
        print("\n" + "="*80)
        print("🎭 EmoSenAI Advanced Text Analysis Results")
        print("="*80)
        
        # Basic info
        print(f"📝 Original Text: {result['text']['original']}")
        print(f"🧼 Cleaned Text: {result['text']['cleaned']}")
        print(f"⏰ Processed: {result['metadata']['timestamp']}")
        
        # Main predictions
        print(f"\n🎯 PREDICTIONS:")
        print(f"   💬 Sentiment: {result['analysis']['sentiment']['predicted']} "
              f"({result['analysis']['sentiment']['confidence']}% confidence)")
        print(f"   🎭 Emotion: {result['analysis']['emotion']['predicted']} "
              f"({result['analysis']['emotion']['confidence']}% confidence)")
        print(f"   ⚠️  Toxicity: {result['analysis']['toxicity']['predicted']} "
              f"({result['analysis']['toxicity']['confidence']}% confidence) "
              f"{'🚨 TOXIC' if result['analysis']['toxicity']['is_toxic'] else '✅ Clean'}")
        
        # Combined metrics
        print(f"\n📊 COMBINED METRICS:")
        print(f"   📈 Overall Score: {result['combined_metrics']['overall_sentiment_score']:.2f}")
        print(f"   🎚️  Emotional Intensity: {result['combined_metrics']['emotional_intensity']:.2f}")
        print(f"   🚦 Risk Level: {result['combined_metrics']['risk_level'].upper()}")
        
        # Insights
        print(f"\n💡 INSIGHTS:")
        for i, insight in enumerate(result['insights'], 1):
            print(f"   {i}. {insight}")
        
        # Text features
        print(f"\n📋 TEXT FEATURES:")
        features = result['text']['features']
        print(f"   📊 Word Count: {features['word_count']}")
        print(f"   📝 Sentence Count: {features['sentence_count']}")
        print(f"   📖 Readability: {features['readability_score']:.1f}")
        print(f"   ❓ Contains Questions: {'Yes' if features['has_questions'] else 'No'}")
        print(f"   ❗ Contains Exclamations: {'Yes' if features['has_exclamations'] else 'No'}")
        
        print("="*80)
    
    def interactive_mode(self):
        """Interactive CLI mode."""
        print("🎭 EmoSenAI Advanced Text Analyzer - Interactive Mode")
        print("Type 'quit' to exit, 'stats' for statistics, 'report' to generate report")
        
        while True:
            try:
                user_input = input("\n📝 Enter text to analyze: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'stats':
                    stats = self.analyzer.get_analysis_stats()
                    print(f"\n📈 Analysis Statistics: {json.dumps(stats, indent=2)}")
                    continue
                elif user_input.lower() == 'report':
                    filename = ReportGenerator.generate_report(self.analyzer)
                    print(f"📄 Report generated: {filename}")
                    continue
                elif not user_input:
                    continue
                
                # Perform analysis
                start_time = time.time()
                result = self.analyzer.comprehensive_analysis(user_input)
                analysis_time = time.time() - start_time
                
                print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
                self.display_results(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using EmoSenAI!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")
    
    def batch_mode(self, file_path: str):
        """Batch analysis mode from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"📁 Processing {len(texts)} texts from {file_path}...")
            results = self.analyzer.batch_analyze(texts)
            
            # Save results
            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Batch analysis complete! Results saved to {output_file}")
            
            # Generate summary
            df = ReportGenerator.to_dataframe(results)
            print(f"\n📊 Summary Statistics:")
            print(f"   Total texts analyzed: {len(df)}")
            print(f"   Average sentiment confidence: {df['sentiment_confidence'].mean():.1f}%")
            print(f"   Most common sentiment: {df['sentiment'].mode().iloc[0]}")
            print(f"   Most common emotion: {df['emotion'].mode().iloc[0]}")
            print(f"   Toxic content: {df['is_toxic'].sum()} texts")
            
        except Exception as e:
            logger.error(f"Batch mode error: {e}")
            print(f"❌ Error processing batch file: {e}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EmoSenAI Advanced Text Analyzer")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Operation mode")
    parser.add_argument("--file", help="Input file for batch mode")
    parser.add_argument("--text", help="Direct text input for analysis")
    
    args = parser.parse_args()
    
    cli = AdvancedCLI()
    
    try:
        if args.text:
            # Direct text analysis
            result = cli.analyzer.comprehensive_analysis(args.text)
            cli.display_results(result)
        elif args.mode == "batch" and args.file:
            # Batch mode
            cli.batch_mode(args.file)
        else:
            # Interactive mode
            cli.interactive_mode()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
    finally:
        # Generate final report
        try:
            ReportGenerator.generate_report(cli.analyzer, "final_report.json")
        except:
            pass