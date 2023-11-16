# -Film-Junky-Union-Movie-Review-Classification-Project
 Film Junky Union Movie Review Classification Project
# Film Junky Union Movie Review Classification Project

## Project Overview

The Film Junky Union, a vibrant community for classic movie enthusiasts, is developing a sophisticated system to filter and categorize movie reviews. The primary objective is to train a model capable of automatically detecting negative reviews within the dataset of IMBD movie reviews. The project aims for the model to achieve an F1 score of at least 0.85.

## Project Statement

1. **Baseline Performance:**
   - The initial DummyClassifier serves as a baseline, producing an F1 score of 50%, equivalent to random chance.

2. **LogisticRegression Models:**
   - Comparative analysis between models using NLTK and Spacy for preprocessing reveals that the NLTK-processed model achieves the best F1 score at 88%. Despite being computationally more efficient (9x lower preprocessing time), the Spacy-processed model finishes training about 8 seconds earlier.

3. **Training Approach:**
   - Both models undergo training using 5-fold cross-validation with hyperparameter tuning on the C parameter.

4. **LightGBM Model:**
   - The LightGBM model exhibits a lower F1 score and a prolonged training time of over an hour.

5. **BERT Embeddings:**
   - Utilizing GPU resources, BERT embeddings are generated within approximately 20 minutes for both training and test reviews. The LogisticRegression model built on these embeddings performs less favorably with an F1 score of 86.4%.

## Conclusions

1. **Evaluation Metrics for BERT Models:**
   - Despite BERT embeddings producing a relatively lower F1 score, the quality of predictions on sample reviews surpasses that of the model with the best F1 score. This suggests that F1 score might not be the most suitable evaluation metric for BERT-trained models.

2. **GPU Utilization:**
   - The BERT embeddings are obtained using the GPU provided in the Colab platform, optimizing computational efficiency.

3. **BERT vs TFIDF:**
   - BERT embeddings outperform TFIDF vectors, capturing the semantic representation of the underlying dataset. Despite longer training times, BERT embeddings offer enhanced predictive capabilities.
