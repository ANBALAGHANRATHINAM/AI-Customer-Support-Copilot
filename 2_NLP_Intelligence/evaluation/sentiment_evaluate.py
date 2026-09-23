#!/usr/bin/env python3
"""
SENTIMENT MODEL EVALUATION SCRIPT
==================================

This script evaluates the existing sentiment model against manually curated messages.

Model: j-hartmann/emotion-english-distilroberta-base
Dataset: Manually labeled ZENDS telecom customer support messages (100 messages)
Labels: Happy, Neutral, Angry

RUN THIS IN YOUR VS CODE TERMINAL:
    python sentiment_evaluate.py --dataset sentiment_manual_test.csv --output sentiment_metrics.json

"""

import csv
import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import pipeline


def load_dataset(csv_path):
    """Load manually labeled evaluation dataset"""
    evaluation_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evaluation_data.append({
                'message': row['message'],
                'ground_truth': row['ground_truth_label']
            })
    return evaluation_data


def run_evaluation(dataset_path, output_path='sentiment_metrics.json'):
    """Run sentiment model evaluation"""
    
    # Load dataset
    print(f"\n{'='*70}")
    print("SENTIMENT MODEL EVALUATION")
    print('='*70)
    
    evaluation_data = load_dataset(dataset_path)
    print(f"\n✓ Loaded {len(evaluation_data)} messages for evaluation")
    
    happy_count = sum(1 for d in evaluation_data if d['ground_truth'] == 'Happy')
    neutral_count = sum(1 for d in evaluation_data if d['ground_truth'] == 'Neutral')
    angry_count = sum(1 for d in evaluation_data if d['ground_truth'] == 'Angry')
    
    print(f"\nGround Truth Distribution:")
    print(f"  Happy:   {happy_count} ({happy_count/len(evaluation_data)*100:.0f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/len(evaluation_data)*100:.0f}%)")
    print(f"  Angry:   {angry_count} ({angry_count/len(evaluation_data)*100:.0f}%)")
    
    # Load the sentiment model
    print("\n→ Loading sentiment model: j-hartmann/emotion-english-distilroberta-base")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    print("✓ Model loaded")
    
    # Emotion to sentiment mapping
    emotion_mapping = {
        'joy': 'Happy',
        'neutral': 'Neutral',
        'surprise': 'Neutral',
        'anger': 'Angry',
        'disgust': 'Angry',
        'fear': 'Angry',
        'sadness': 'Angry'
    }
    
    # Run inference
    print("\n→ Running inference on all 100 messages...")
    predictions_mapped = []
    ground_truth = []
    
    for i, data in enumerate(evaluation_data):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/100]", end="\r")
        
        message = data['message']
        ground_truth_label = data['ground_truth']
        
        # Run model
        result = classifier(message)
        
        # Extract emotion with highest score
        # Note: with top_k=None, result is a list containing a list of dicts
        # result[0] contains the actual emotion predictions
        emotions_scores = {item['label']: item['score'] for item in result[0]}
        top_emotion = max(emotions_scores, key=emotions_scores.get)
        mapped_sentiment = emotion_mapping.get(top_emotion, 'Neutral')
        
        predictions_mapped.append(mapped_sentiment)
        ground_truth.append(ground_truth_label)
    
    print(f"  [100/100] ✓ Inference complete")
    
    # Calculate metrics
    print("\n→ Calculating metrics...")
    
    accuracy = accuracy_score(ground_truth, predictions_mapped)
    precision_weighted = precision_score(ground_truth, predictions_mapped, average='weighted', zero_division=0)
    recall_weighted = recall_score(ground_truth, predictions_mapped, average='weighted', zero_division=0)
    f1_weighted = f1_score(ground_truth, predictions_mapped, average='weighted', zero_division=0)
    
    class_report = classification_report(ground_truth, predictions_mapped, output_dict=True, zero_division=0)
    
    labels_list = ['Angry', 'Happy', 'Neutral']
    cm = confusion_matrix(ground_truth, predictions_mapped, labels=labels_list)
    cm_list = cm.tolist()
    
    # Create results
    results = {
        "model": "j-hartmann/emotion-english-distilroberta-base",
        "evaluation_type": "manual_curated_telecom_messages",
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "test_examples": len(evaluation_data),
        "ground_truth_distribution": {
            "Happy": happy_count,
            "Neutral": neutral_count,
            "Angry": angry_count
        },
        "emotion_mapping": {
            "Happy": ["joy"],
            "Neutral": ["neutral", "surprise"],
            "Angry": ["anger", "disgust", "fear", "sadness"]
        },
        "confusion_matrix": {
            "labels": ["Angry", "Happy", "Neutral"],
            "matrix": cm_list
        },
        "classification_report": {
            "Angry": {
                "precision": float(class_report.get("Angry", {}).get("precision", 0)),
                "recall": float(class_report.get("Angry", {}).get("recall", 0)),
                "f1-score": float(class_report.get("Angry", {}).get("f1-score", 0)),
                "support": int(class_report.get("Angry", {}).get("support", 0))
            },
            "Happy": {
                "precision": float(class_report.get("Happy", {}).get("precision", 0)),
                "recall": float(class_report.get("Happy", {}).get("recall", 0)),
                "f1-score": float(class_report.get("Happy", {}).get("f1-score", 0)),
                "support": int(class_report.get("Happy", {}).get("support", 0))
            },
            "Neutral": {
                "precision": float(class_report.get("Neutral", {}).get("precision", 0)),
                "recall": float(class_report.get("Neutral", {}).get("recall", 0)),
                "f1-score": float(class_report.get("Neutral", {}).get("f1-score", 0)),
                "support": int(class_report.get("Neutral", {}).get("support", 0))
            },
            "accuracy": float(accuracy),
            "macro avg": {
                "precision": float(class_report.get("macro avg", {}).get("precision", 0)),
                "recall": float(class_report.get("macro avg", {}).get("recall", 0)),
                "f1-score": float(class_report.get("macro avg", {}).get("f1-score", 0)),
                "support": float(class_report.get("macro avg", {}).get("support", 0))
            },
            "weighted avg": {
                "precision": float(class_report.get("weighted avg", {}).get("precision", 0)),
                "recall": float(class_report.get("weighted avg", {}).get("recall", 0)),
                "f1-score": float(class_report.get("weighted avg", {}).get("f1-score", 0)),
                "support": float(class_report.get("weighted avg", {}).get("support", 0))
            }
        }
    }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Display results
    print(f"\n{'EVALUATION RESULTS':^70}")
    print('='*70)
    
    print(f"\nModel:           j-hartmann/emotion-english-distilroberta-base")
    print(f"Dataset:         Manually curated ZENDS telecom messages")
    print(f"Test Samples:    {len(evaluation_data)}")
    
    print(f"\n{'OVERALL METRICS':^70}")
    print('-'*70)
    print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision (weighted):  {precision_weighted:.4f}")
    print(f"Recall (weighted):     {recall_weighted:.4f}")
    print(f"F1-Score (weighted):   {f1_weighted:.4f}")
    
    print(f"\n{'PER-CLASS METRICS':^70}")
    print('-'*70)
    for sentiment in ['Happy', 'Neutral', 'Angry']:
        if sentiment in class_report:
            metrics = class_report[sentiment]
            print(f"\n{sentiment}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {int(metrics['support'])}")
    
    print(f"\n{'CONFUSION MATRIX':^70}")
    print('-'*70)
    print("\nPredicted →")
    print("Actual ↓       Angry    Happy   Neutral")
    for i, true_label in enumerate(['Angry', 'Happy', 'Neutral']):
        row = cm[i]
        print(f"{true_label:10s}  {row[0]:5d}   {row[1]:5d}    {row[2]:5d}")
    
    print(f"\n{'='*70}\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate sentiment model on manually curated test set'
    )
    parser.add_argument(
        '--dataset',
        default='sentiment_manual_test.csv',
        help='Path to evaluation dataset CSV'
    )
    parser.add_argument(
        '--output',
        default='sentiment_metrics.json',
        help='Path to save metrics JSON'
    )
    
    args = parser.parse_args()
    
    run_evaluation(args.dataset, args.output)
