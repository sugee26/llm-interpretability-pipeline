"""
Example: Sentiment Classification with Interpretability

This script demonstrates how to use the LLM Interpretability Pipeline
for sentiment classification with full interpretability support.
"""

import sys
sys.path.insert(0, "..")

from src.pipeline import InterpretableNLPPipeline


def main():
    # Sample training data
    train_texts = [
        "This product is amazing! Best purchase ever.",
        "Terrible quality, broke after one day.",
        "Really happy with my order, works perfectly.",
        "Waste of money, do not buy this.",
        "Exceeded my expectations, highly recommend!",
        "Poor customer service and defective item.",
        "Love it! Will definitely buy again.",
        "Disappointed, not as described.",
        "Fantastic product, great value for money.",
        "Horrible experience, want my money back.",
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = InterpretableNLPPipeline(
        model_name="distilbert-base-uncased",
        num_labels=2,
        interpretability_methods=["shap", "lime", "attention"],
        max_length=128,
    )

    # Train the model
    print("\nTraining model...")
    history = pipeline.fit(
        train_texts=train_texts,
        train_labels=train_labels,
        epochs=3,
        batch_size=4,
        label_names=["Negative", "Positive"],
    )

    print(f"\nTraining complete! Final loss: {history['history'][-1]['train_loss']:.4f}")

    # Test predictions
    test_texts = [
        "Absolutely love this product!",
        "Terrible, don't waste your money.",
        "It's okay, nothing special.",
    ]

    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    predictions, probs = pipeline.predict(test_texts, return_probs=True)
    for text, pred, prob in zip(test_texts, predictions, probs):
        label = pipeline.label_names[pred]
        confidence = prob[pred] * 100
        print(f"\nText: {text}")
        print(f"Prediction: {label} ({confidence:.1f}% confidence)")

    # Generate explanations
    print("\n" + "=" * 60)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # Analyze a positive example
    positive_text = "This is the best product I have ever bought!"
    print(f"\nAnalyzing: '{positive_text}'")

    # SHAP explanation
    print("\n--- SHAP Analysis ---")
    shap_exp = pipeline.explain_shap(positive_text, num_samples=50, visualize=False)
    print(f"Predicted: {pipeline.label_names[shap_exp['predicted_class']]}")
    print("Top contributing words:")
    for attr in shap_exp["token_attributions"][:5]:
        sign = "+" if attr["attribution"] > 0 else ""
        print(f"  {attr['token']}: {sign}{attr['attribution']:.4f}")

    # LIME explanation
    print("\n--- LIME Analysis ---")
    lime_exp = pipeline.explain_lime(positive_text, num_features=5, visualize=False)
    print("Feature weights:")
    for fw in lime_exp["feature_weights"]:
        sign = "+" if fw["weight"] > 0 else ""
        print(f"  '{fw['word']}': {sign}{fw['weight']:.4f}")

    # Attention analysis
    print("\n--- Attention Analysis ---")
    attn_exp = pipeline.explainers["attention"].explain(positive_text)
    print("Top attended tokens:")
    sorted_attrs = sorted(
        attn_exp["token_attributions"],
        key=lambda x: x["importance"],
        reverse=True,
    )
    for attr in sorted_attrs[:5]:
        if attr["token"] not in ["[CLS]", "[SEP]", "[PAD]"]:
            print(f"  {attr['token']}: {attr['importance']:.4f}")

    # Save model
    print("\n" + "=" * 60)
    print("Saving model to './outputs/sentiment_model'...")
    pipeline.save("./outputs/sentiment_model")
    print("Done!")


if __name__ == "__main__":
    main()
