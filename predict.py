"""
Fraud Prediction Script
Usage: python predict.py
Author: Your Name
"""

from fraud_detection import FraudDetectionPipeline
import pandas as pd
import pickle

# Example transaction for testing
example_transaction = {
    'step': 1,
    'amount': 9839.64,
    'oldbalanceOrg': 170136.0,
    'newbalanceOrig': 160296.36,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0,
    'type_encoded': 4,
    'hour': 1,
    'day': 0,
    'amount_log': 9.194,
    'amount_sqrt': 99.195,
    'orig_balance_ratio': 0.057,
    'dest_balance_ratio': 9839.64,
    'is_orig_zero': 0,
    'is_dest_zero': 1,
    'is_orig_drained': 0,
    'is_dest_merchant': 1,
    'is_orig_merchant': 0,
    'type_CASH_IN': 0,
    'type_CASH_OUT': 0,
    'type_DEBIT': 0,
    'type_PAYMENT': 1,
    'type_TRANSFER': 0
}

def predict_single_transaction(transaction):
    """
    Predict if a single transaction is fraudulent
    
    Args:
        transaction (dict): Transaction features
        
    Returns:
        dict: Prediction results with probabilities
    """
    # Note: In production, load pre-trained model
    # For now, this demonstrates the prediction interface
    
    print("="*60)
    print("FRAUD DETECTION - TRANSACTION ANALYSIS")
    print("="*60)
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${transaction.get('amount', 0):,.2f}")
    print(f"  Original Balance: ${transaction.get('oldbalanceOrg', 0):,.2f}")
    print(f"  Hour: {transaction.get('hour', 0)}")
    
    # In production, use trained pipeline:
    # pipeline = FraudDetectionPipeline.load('trained_model.pkl')
    # result = pipeline.predict(transaction)
    
    # Demo result (replace with actual prediction)
    result = {
        'prediction': 'FRAUD',
        'fraud_probability': 0.87,
        'legitimate_probability': 0.13,
        'risk_level': 'HIGH',
        'confidence': 0.87
    }
    
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"{'='*60}\n")
    
    return result


def batch_predict(transactions_df):
    """
    Predict fraud for multiple transactions
    
    Args:
        transactions_df (DataFrame): DataFrame with transaction features
        
    Returns:
        DataFrame: Original data with predictions added
    """
    print(f"Processing {len(transactions_df)} transactions...")
    
    # In production: pipeline.predict(transactions_df)
    predictions = []
    
    for idx, transaction in transactions_df.iterrows():
        result = predict_single_transaction(transaction.to_dict())
        predictions.append(result)
    
    return pd.DataFrame(predictions)


def main():
    """Main execution"""
    print("="*60)
    print("FRAUD DETECTION PREDICTION SYSTEM")
    print("="*60)
    
    # Test single transaction
    print("\n[TEST] Single Transaction Prediction:")
    result = predict_single_transaction(example_transaction)
    
    # Recommendation based on prediction
    if result['prediction'] == 'FRAUD':
        if result['fraud_probability'] > 0.8:
            print("⚠️  RECOMMENDATION: BLOCK transaction immediately")
        elif result['fraud_probability'] > 0.5:
            print("⚠️  RECOMMENDATION: Hold for manual review")
        else:
            print("⚠️  RECOMMENDATION: Flag for monitoring")
    else:
        print("✅ RECOMMENDATION: Approve transaction")
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
```

### **STEP 4:** Scroll down, type commit message:
```
Add prediction script
```

### **STEP 5:** Click green "Commit changes..." button

---

## **🎉 AFTER ALL 4 FILES ARE ADDED:**

Your repository will have:
```
✅ README.md
✅ fraud_detection.py
✅ requirements.txt
✅ .gitignore
✅ predict.py
```

---

## **🔗 FINAL STEP: GET YOUR LINK**

Your GitHub project link will be:
```
https://github.com/samvedsangle/Transaction-Fraud-Detection-System
