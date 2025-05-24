import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

class EmailDataScraper:
    """
    A class to scrape email data from various sources for spam/ham classification
    """
    
    def __init__(self):
        self.ham_emails = []
        self.spam_emails = []
        
    def scrape_sample_emails(self):
        """
        Generate sample email data for demonstration purposes
        In a real scenario, you would scrape from legitimate sources
        """
        # Sample ham emails
        ham_samples = [
            "Hi John, thanks for your help with the project. The meeting is scheduled for tomorrow at 3 PM.",
            "Don't forget about dinner tonight at 7 PM. Looking forward to seeing you!",
            "Your monthly statement is ready for review. Please log in to your account.",
            "Team meeting moved to Friday 10 AM. Please update your calendars accordingly.",
            "Happy birthday! Hope you have a wonderful day filled with joy and celebration.",
            "The quarterly report has been submitted successfully. Thank you for your contributions.",
            "Reminder: Your appointment is scheduled for next Tuesday at 2 PM.",
            "Great job on the presentation today. The client was very impressed.",
            "Your order has been shipped and will arrive within 3-5 business days.",
            "Please review the attached document and provide feedback by end of week."
        ]
        
        # Sample spam emails
        spam_samples = [
            "URGENT: You've won $1,000,000! Click here to claim your prize NOW!",
            "Limited time offer! Get 90% OFF on all products! Act fast!",
            "Your account will be closed unless you verify immediately. Click here!",
            "FREE money waiting for you! No strings attached! Claim now!",
            "Make $5000 per week working from home! No experience required!",
            "WINNER WINNER! You're our lucky customer! Claim your free iPhone!",
            "Your PayPal account has been suspended. Verify now to avoid closure!",
            "Hot singles in your area want to meet you! Join now for FREE!",
            "Get rich quick with this amazing investment opportunity! Guaranteed returns!",
            "URGENT: Your package is stuck in customs. Pay fee to release it!"
        ]
        
        # Add more samples to create imbalanced dataset (more ham than spam)
        additional_ham = [
            "Meeting agenda attached. Please review before tomorrow's session.",
            "Your subscription renewal is due next month. Auto-pay is enabled.",
            "Thank you for your purchase. Receipt attached for your records.",
            "Weather update: Rain expected tomorrow. Drive safely!",
            "Your flight check-in is now available. Seat 14A confirmed.",
            "Library books are due next Friday. Renewal option available online.",
            "Your medication refill is ready for pickup at the pharmacy.",
            "Grocery list: milk, bread, eggs, and vegetables for this week.",
            "Car service appointment confirmed for Saturday morning at 9 AM.",
            "Your test results are normal. Follow-up appointment recommended in 6 months."
        ]
        
        self.ham_emails.extend(ham_samples + additional_ham)
        self.spam_emails.extend(spam_samples)
        
        return len(self.ham_emails), len(self.spam_emails)
    
    def create_dataset(self, filename='email_dataset.csv'):
        """
        Create a CSV dataset from scraped emails
        """
        # Create DataFrame
        ham_df = pd.DataFrame({
            'label': ['ham'] * len(self.ham_emails),
            'text': self.ham_emails
        })
        
        spam_df = pd.DataFrame({
            'label': ['spam'] * len(self.spam_emails),
            'text': self.spam_emails
        })
        
        # Combine and shuffle
        df = pd.concat([ham_df, spam_df], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Dataset created: {filename}")
        print(f"Total emails: {len(df)}")
        print(f"Ham emails: {len(ham_df)}")
        print(f"Spam emails: {len(spam_df)}")
        
        return df

if __name__ == "__main__":
    scraper = EmailDataScraper()
    ham_count, spam_count = scraper.scrape_sample_emails()
    print(f"Scraped {ham_count} ham emails and {spam_count} spam emails")
    
    dataset = scraper.create_dataset()
    print("\nDataset preview:")
    print(dataset.head(10))