import smtplib
import json
import logging
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import pandas as pd
import os

class SpamAlertSystem:
    """
    Alert system for spam detection monitoring and notifications
    """
    
    def __init__(self, config_file='alert_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.alert_log = []
        self.setup_logging()
        
    def load_config(self):
        """
        Load alert system configuration
        """
        default_config = {
            "email_settings": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "recipient_emails": []
            },
            "alert_thresholds": {
                "spam_rate_threshold": 0.1,  # Alert if spam rate > 10%
                "confidence_threshold": 0.8,  # Alert for low confidence predictions
                "daily_spam_limit": 100,      # Alert if daily spam > 100
                "consecutive_spam_limit": 10   # Alert for consecutive spam
            },
            "monitoring": {
                "enabled": True,
                "log_file": "spam_alerts.log",
                "alert_cooldown": 3600  # 1 hour cooldown between similar alerts
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in loaded_config:
                        loaded_config[key] = default_config[key]
                return loaded_config
            else:
                # Create default config file
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return default_config
    
    def setup_logging(self):
        """
        Setup logging for alert system
        """
        log_file = self.config['monitoring']['log_file']
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def send_email_alert(self, subject, message, alert_type="SPAM_DETECTION"):
        """
        Send email alert
        """
        if not self.config['email_settings']['sender_email']:
            print("Email not configured. Alert logged only.")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.config['email_settings']['sender_email']
            msg['To'] = ', '.join(self.config['email_settings']['recipient_emails'])
            msg['Subject'] = f"[{alert_type}] {subject}"
            
            # Email body
            body = f"""
SPAM DETECTION ALERT
==================

Alert Type: {alert_type}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}

---
This is an automated alert from the Spam Detection System.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                self.config['email_settings']['smtp_server'],
                self.config['email_settings']['smtp_port']
            )
            server.starttls()
            server.login(
                self.config['email_settings']['sender_email'],
                self.config['email_settings']['sender_password']
            )
            
            text = msg.as_string()
            server.sendmail(
                self.config['email_settings']['sender_email'],
                self.config['email_settings']['recipient_emails'],
                text
            )
            server.quit()
            
            self.logger.info(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def check_spam_rate_alert(self, recent_predictions):
        """
        Check if spam rate exceeds threshold
        """
        if len(recent_predictions) < 10:  # Need minimum samples
            return False
        
        spam_count = sum(1 for pred in recent_predictions if pred['prediction'] == 'spam')
        spam_rate = spam_count / len(recent_predictions)
        
        threshold = self.config['alert_thresholds']['spam_rate_threshold']
        
        if spam_rate > threshold:
            subject = f"High Spam Rate Alert: {spam_rate:.2%}"
            message = f"""
High spam detection rate detected!

Statistics:
- Total emails processed: {len(recent_predictions)}
- Spam emails detected: {spam_count}
- Spam rate: {spam_rate:.2%}
- Threshold: {threshold:.2%}

This may indicate:
1. Increase in spam volume
2. Potential system issues
3. Need for model retraining
            """
            
            self.send_email_alert(subject, message, "HIGH_SPAM_RATE")
            self.logger.warning(f"High spam rate alert: {spam_rate:.2%}")
            return True
        
        return False
    
    def check_low_confidence_alert(self, prediction_result):
        """
        Check for low confidence predictions
        """
        if prediction_result.get('confidence'):
            confidence = prediction_result['confidence']
            threshold = self.config['alert_thresholds']['confidence_threshold']
            
            if confidence < threshold:
                subject = f"Low Confidence Prediction: {confidence:.2%}"
                message = f"""
Low confidence prediction detected!

Prediction Details:
- Text: {prediction_result.get('text', 'N/A')[:200]}...
- Prediction: {prediction_result['prediction']}
- Confidence: {confidence:.2%}
- Threshold: {threshold:.2%}

This may require manual review.
                """
                
                self.send_email_alert(subject, message, "LOW_CONFIDENCE")
                self.logger.warning(f"Low confidence prediction: {confidence:.2%}")
                return True
        
        return False
    
    def check_consecutive_spam_alert(self, recent_predictions, limit=None):
        """
        Check for consecutive spam detections
        """
        if limit is None:
            limit = self.config['alert_thresholds']['consecutive_spam_limit']
        
        consecutive_count = 0
        for pred in reversed(recent_predictions):
            if pred['prediction'] == 'spam':
                consecutive_count += 1
            else:
                break
        
        if consecutive_count >= limit:
            subject = f"Consecutive Spam Alert: {consecutive_count} in a row"
            message = f"""
Multiple consecutive spam emails detected!

Details:
- Consecutive spam count: {consecutive_count}
- Alert threshold: {limit}

Recent spam emails:
            """
            
            for i, pred in enumerate(recent_predictions[-consecutive_count:]):
                message += f"\n{i+1}. {pred.get('text', 'N/A')[:100]}..."
            
            self.send_email_alert(subject, message, "CONSECUTIVE_SPAM")
            self.logger.warning(f"Consecutive spam alert: {consecutive_count} emails")
            return True
        
        return False
    
    def monitor_daily_stats(self, predictions_today):
        """
        Monitor daily spam statistics
        """
        spam_count = sum(1 for pred in predictions_today if pred['prediction'] == 'spam')
        limit = self.config['alert_thresholds']['daily_spam_limit']
        
        if spam_count > limit:
            subject = f"Daily Spam Limit Exceeded: {spam_count} emails"
            message = f"""
Daily spam limit exceeded!

Statistics:
- Spam emails today: {spam_count}
- Daily limit: {limit}
- Total emails processed: {len(predictions_today)}
- Spam rate: {spam_count/len(predictions_today):.2%}

Consider investigating potential spam campaigns or system issues.
            """
            
            self.send_email_alert(subject, message, "DAILY_LIMIT")
            self.logger.warning(f"Daily spam limit exceeded: {spam_count}")
            return True
        
        return False
    
    def generate_daily_report(self, predictions_today):
        """
        Generate daily summary report
        """
        if not predictions_today:
            return
        
        total_emails = len(predictions_today)
        spam_count = sum(1 for pred in predictions_today if pred['prediction'] == 'spam')
        ham_count = total_emails - spam_count
        spam_rate = spam_count / total_emails if total_emails > 0 else 0
        
        # Calculate average confidence
        confidences = [pred.get('confidence', 0) for pred in predictions_today if pred.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        subject = f"Daily Spam Detection Report - {datetime.now().strftime('%Y-%m-%d')}"
        message = f"""
DAILY SPAM DETECTION SUMMARY
============================

Date: {datetime.now().strftime('%Y-%m-%d')}

Statistics:
- Total emails processed: {total_emails}
- Spam emails detected: {spam_count}
- Ham emails detected: {ham_count}
- Spam rate: {spam_rate:.2%}
- Average confidence: {avg_confidence:.2%}

Performance Metrics:
- High confidence predictions: {sum(1 for c in confidences if c > 0.9)}
- Medium confidence predictions: {sum(1 for c in confidences if 0.7 <= c <= 0.9)}
- Low confidence predictions: {sum(1 for c in confidences if c < 0.7)}

Top spam indicators detected today:
        """
        
        # Add top spam phrases if available
        spam_texts = [pred.get('text', '') for pred in predictions_today if pred['prediction'] == 'spam']
        if spam_texts:
            spam_words = {}
            for text in spam_texts:
                words = text.lower().split()
                for word in words:
                    if len(word) > 3:  # Only count meaningful words
                        spam_words[word] = spam_words.get(word, 0) + 1
            
            top_words = sorted(spam_words.items(), key=lambda x: x[1], reverse=True)[:10]
            for word, count in top_words:
                message += f"\n- '{word}': {count} occurrences"
        
        self.send_email_alert(subject, message, "DAILY_REPORT")
        self.logger.info(f"Daily report generated: {total_emails} emails processed")
    
    def process_prediction(self, text, prediction_result, recent_predictions=None):
        """
        Process a single prediction and check for alerts
        """
        if not self.config['monitoring']['enabled']:
            return
        
        # Add timestamp to prediction
        prediction_result['timestamp'] = datetime.now().isoformat()
        prediction_result['text'] = text
        
        # Log prediction
        self.logger.info(f"Prediction: {prediction_result['prediction']} (confidence: {prediction_result.get('confidence', 'N/A')})")
        
        # Check various alert conditions
        alerts_triggered = []
        
        # Low confidence alert
        if self.check_low_confidence_alert(prediction_result):
            alerts_triggered.append('low_confidence')
        
        # Check recent predictions for patterns
        if recent_predictions:
            # High spam rate alert
            if self.check_spam_rate_alert(recent_predictions):
                alerts_triggered.append('high_spam_rate')
            
            # Consecutive spam alert
            if self.check_consecutive_spam_alert(recent_predictions):
                alerts_triggered.append('consecutive_spam')
        
        return alerts_triggered

class PredictionLogger:
    """
    Logger for keeping track of predictions for alert system
    """
    
    def __init__(self, log_file='predictions.json', max_entries=1000):
        self.log_file = log_file
        self.max_entries = max_entries
        self.predictions = self.load_predictions()
    
    def load_predictions(self):
        """
        Load existing predictions from file
        """
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return []
    
    def save_predictions(self):
        """
        Save predictions to file
        """
        try:
            # Keep only recent entries
            if len(self.predictions) > self.max_entries:
                self.predictions = self.predictions[-self.max_entries:]
            
            with open(self.log_file, 'w') as f:
                json.dump(self.predictions, f, indent=2)
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def add_prediction(self, text, prediction_result):
        """
        Add new prediction to log
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text[:200],  # Truncate long texts
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result.get('confidence'),
            'raw_prediction': prediction_result.get('raw_prediction')
        }
        
        self.predictions.append(entry)
        self.save_predictions()
    
    def get_recent_predictions(self, hours=24):
        """
        Get predictions from last N hours
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for pred in self.predictions:
            try:
                pred_time = datetime.fromisoformat(pred['timestamp'])
                if pred_time > cutoff_time:
                    recent.append(pred)
            except:
                continue
        
        return recent
    
    def get_daily_stats(self, date=None):
        """
        Get statistics for a specific date
        """
        if date is None:
            date = datetime.now().date()
        
        daily_preds = []
        for pred in self.predictions:
            try:
                pred_date = datetime.fromisoformat(pred['timestamp']).date()
                if pred_date == date:
                    daily_preds.append(pred)
            except:
                continue
        
        return daily_preds

# Example usage function
def setup_alert_system():
    """
    Setup and configure the alert system
    """
    # Create alert system
    alert_system = SpamAlertSystem()
    prediction_logger = PredictionLogger()
    
    # Example configuration
    config = {
        "email_settings": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",  # Configure this
            "sender_password": "your-app-password",   # Configure this
            "recipient_emails": ["admin@example.com"] # Configure this
        },
        "alert_thresholds": {
            "spam_rate_threshold": 0.15,
            "confidence_threshold": 0.75,
            "daily_spam_limit": 50,
            "consecutive_spam_limit": 5
        }
    }
    
    # Save configuration
    with open('alert_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Alert system configured!")
    print("Please update alert_config.json with your email settings.")
    
    return alert_system, prediction_logger

if __name__ == "__main__":
    # Setup example
    alert_system, prediction_logger = setup_alert_system()
    
    # Example usage
    sample_predictions = [
        {'prediction': 'spam', 'confidence': 0.95},
        {'prediction': 'spam', 'confidence': 0.87},
        {'prediction': 'ham', 'confidence': 0.92},
        {'prediction': 'spam', 'confidence': 0.65},  # Low confidence
    ]
    
    for i, pred in enumerate(sample_predictions):
        text = f"Sample email text {i+1}"
        prediction_logger.add_prediction(text, pred)
        alert_system.process_prediction(text, pred, prediction_logger.get_recent_predictions())
    
    print("Alert system demo completed!")