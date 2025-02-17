# Sentiment Analysis of Movie Reviews for “Paddington in Peru”
**Project Report**

## 1. Introduction

This project aimed to analyze public sentiment towards the movie "Paddington in Peru" using movie reviews from various online sources. Sentiment analysis, the process of determining the emotional tone behind text, was employed to gauge overall audience reception. The project leveraged a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model for sentiment classification and incorporated web scraping techniques to gather review data from platforms like Rotten Tomatoes, IMDB, BBC, and Reddit.

### 1.1 Objectives

*   Fine-tune a pre-trained BERT model for sentiment classification using a large dataset of movie reviews.
*   Develop web scraping functionality to collect reviews for "Paddington in Peru" from multiple sources.
*   Perform sentiment analysis on the collected reviews using the fine-tuned model.
*   Analyze the results to determine the overall sentiment towards the movie.

## 2. Data Importing, Preprocessing, and Tokenization

### 2.1 Data Importing

The project commenced by mounting Google Drive to access necessary files and uploading the Kaggle API key for dataset download. The IMDB Dataset of 50K Movie Reviews was imported to serve as the training data for fine-tuning the sentiment analysis model. Initially, a smaller subset of the data (1000 training samples and 250 testing samples) was used for initial development and experimentation. This was done to accelerate the development cycle and identify potential issues early on. However, it quickly became apparent that this smaller dataset, while saving considerable processing time, presented challenges in achieving satisfactory model accuracy and balancing the risk of overfitting. Consequently, the decision was made to utilize the entire IMDB dataset with a standard 80/20 train-test split. This larger dataset provided the model with more information to learn from, leading to improved performance.

### 2.2 Preprocessing

The 'sentiment' column of the IMDB dataset, containing text labels ("positive" and "negative"), was converted to numerical representations (1 and 0, respectively) for compatibility with the machine learning model. The pandas DataFrames `train_df` and `test_df`, resulting from the train-test split, were then converted into the Hugging Face `Dataset` format using the `Dataset.from_pandas` function. This conversion is essential for seamless integration with the Hugging Face `transformers` library and its model training tools. The `Dataset` object provides efficient data handling and access during the training process.

### 2.3 Tokenization

A crucial step in preparing the data for the BERT model was tokenization. A function named `tokenize_function` was defined to handle this process. This function takes a batch of movie reviews as input and performs the following operations:

*   **Tokenization:** Each review is broken down into individual words or sub-word units (tokens) using the BERT tokenizer. This converts the text into numerical IDs that the BERT model can understand.
*   **Padding/Truncation:** BERT models require input sequences of a fixed length. Therefore, reviews shorter than the maximum sequence length are padded with special tokens to make them the same length, while reviews longer than the maximum length are truncated. This ensures consistency in input size for the model.

The output of the `tokenize_function` is a dictionary containing the token IDs, attention masks (indicating which tokens are actual words and which are padding), and labels (the sentiment of the review). This tokenized data is then used as input for training and evaluating the sentiment classification model.

## 3. Model

### 3.1 Model Selection and Justification

#### 3.1.1 Traditional Machine Learning Considerations

Before delving into neural networks and specifically BERT, several traditional machine learning algorithms and techniques were considered for sentiment analysis. These included:

*   **Naive Bayes:** A probabilistic classifier based on Bayes' theorem, often used as a baseline for text classification tasks. While computationally efficient, Naive Bayes often makes simplifying assumptions about word independence, which can limit its accuracy.
*   **Support Vector Machines (SVMs):** Powerful algorithms that can effectively separate data points into different classes by finding an optimal hyperplane. SVMs are effective in high-dimensional spaces like text data but can be computationally expensive, especially with large datasets.
*   **Logistic Regression:** A linear model used for binary classification. It's relatively simple and interpretable but may not capture complex non-linear relationships in text data.
*   **Rule-based Systems:** These systems rely on predefined rules to determine sentiment. While easy to understand, they can be brittle and difficult to maintain as language evolves.

While these methods offer viable solutions for sentiment analysis, they often struggle to capture the nuances and complexities of human language. Deep learning models, particularly those based on the Transformer architecture, have demonstrated superior performance in natural language processing tasks, including sentiment analysis. Therefore, the project focused on utilizing BERT.

#### 3.1.2 BERT Variants and Selection

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that has revolutionized NLP. Several BERT variants exist, each with different sizes and performance characteristics. The author explored several BERT variants:

*   **BERT-Base:** The original BERT model, powerful but computationally demanding. Initial attempts to use BERT-Base were hampered by resource constraints on the available hardware (personal laptop), necessitating a shift to lighter models.
*   **BERT-Large:** An even larger version of BERT, offering increased capacity but requiring significantly more computational resources, making it unsuitable for the project's constraints.
*   **DistilBERT:** A distilled version of BERT-Base, offering a good balance between performance and efficiency. While more manageable than BERT-Base, training times were still longer than desired.
*   **TinyBERT:** An extremely compact BERT model designed for resource-constrained environments. While training times were very fast, TinyBERT's performance was insufficient, struggling to find a balance between underfitting and overfitting.

After evaluating these options, the "google/bert_uncased_L-2_H-128_A-2" model was selected. This model represented a "middle ground" – smaller and faster than DistilBERT but larger and more powerful than TinyBERT. This choice reflects a pragmatic approach, balancing the need for reasonable training times with the desire for good model performance.

The chosen BERT model was loaded and initialized using the Hugging Face `transformers` library.  The `model_name` stores the identifier for the pre-trained BERT model. The `tokenizer` loads the appropriate tokenizer, essential for converting text into numerical input. The `model` loads the pre-trained BERT model, configured for sequence classification with `num_labels=2` for binary classification. The model is initialized with pre-trained weights, which are then fine-tuned on the movie review dataset.

## 3.2 Model Fine-tuning

The process of fine-tuning the pre-trained BERT model involves adjusting its parameters on the movie review dataset to specialize it for sentiment analysis. This is accomplished using the Hugging Face `Trainer` class and the `TrainingArguments` class to configure the training process.

### 3.2.1 Training Arguments

The `TrainingArguments` class allows for setting various hyperparameters that govern the training procedure. These hyperparameters significantly influence the model's performance and training dynamics. The following hyperparameters were used:

*   `output_dir`: Specifies the directory on Google Drive where the trained model and training logs are saved.
*   `num_train_epochs`: Sets the number of times the model iterates over the entire training dataset.
*   `per_device_train_batch_size`: Determines the number of training examples processed simultaneously by the model.
*   `per_device_eval_batch_size`: Sets the batch size for evaluating the model's performance.
*   `eval_strategy` and `save_strategy`: Both set to "steps," meaning evaluation and model saving occur at specified intervals.
*   `save_steps` and `eval_steps`: Set to 200, indicating that the model is saved and evaluated every 200 training steps.
*   `learning_rate`: Controls the rate at which the model's weights are updated during training.
*   `weight_decay`: A regularization technique used to prevent overfitting.
*   `load_best_model_at_end`: Ensures that the model with the best performance is loaded at the end of training.
*   `report_to`: Set to "none" in the original code.

### 3.2.2 Trainer Initialization and Training

The `Trainer` class orchestrates the model fine-tuning process. It takes the following arguments:

*   `model`: The pre-trained BERT model.
*   `args`: The `TrainingArguments` object.
*   `train_dataset`: The tokenized training dataset.
*   `eval_dataset`: The tokenized validation dataset.
*   `compute_metrics`: A function used to calculate evaluation metrics.

The `Trainer` is initialized with these arguments, and the `trainer.train()` method is called to begin the fine-tuning process.

#### 3.2.3 Training Execution

The `trainer.train()` method initiated the fine-tuning process. Due to limited resources, the author opted for 3 epochs, despite observing better performance with 20 epochs and a learning rate of 2e-5 in previous tests.

### 3.3 Model Validation and Performance Evaluation

The model's performance was evaluated on the Rotten Tomatoes dataset.

#### 3.3.1 Validation Process

Predictions and labels were stored in lists. The code iterated through the tokenized Rotten Tomatoes dataset, made predictions, and stored the results.

#### 3.3.2 Performance Analysis

*   Accuracy: 0.741
*   Precision: 0.788
*   Recall: 0.660
*   F1-score: 0.718

#### 3.3.3 Interpretation and Potential Improvements

The model achieved reasonable accuracy and precision, but recall could be improved. Potential improvements include more training data, hyperparameter tuning, exploring different model architectures, data augmentation, and error analysis.

## 4. Web Scraping and Sentiment Analysis for "Paddington in Peru"

This section focuses on gathering reviews for "Paddington in Peru" and analyzing their sentiment.

### 4.1 Web Scraping

Reviews were scraped from Rotten Tomatoes, IMDB, BBC, and Reddit using `requests` and Beautiful Soup.  Reddit scraping involved extracting JSON data from script tags.

### 4.2 Sentiment Prediction and Aggregation

The scraped reviews were preprocessed, tokenized, and fed into the fine-tuned BERT model. The overall sentiment was determined by aggregating the individual sentiment predictions.

### 4.3 Results Output

The overall sentiment towards "Paddington in Peru" was printed to the console.

## 5. Conclusion

This project successfully demonstrated sentiment analysis using a fine-tuned BERT model and web scraping. The model achieved reasonable performance. Future work could explore further improvements.  The project provides a useful framework for analyzing public opinion towards movies.

## Sources

*   Lan, Zhenghao, et al. "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models." *arXiv preprint arXiv:1908.08962* (2019). [https://arxiv.org/abs/1908.08962](https://arxiv.org/abs/1908.08962)
*   Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017). [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   Devlin, Jacob, Ming-Wei Chang, Lee, and Kristina Toutanova. "BERT: Pre-training of deep bidirectional transformers for language understanding." *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. 2019. [https://aclanthology.org/N19-1423/](https://aclanthology.org/N19-1423/)
*   Pang, Bo, and Lillian Lee. "Opinion mining and sentiment analysis." *Foundations and Trends® in Information Retrieval* 2.1–2 (2008): 1-135. [https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf](https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)
*   Kim, Yoon. "Convolutional neural networks for sentence classification." *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 2014. [https://www.aclweb.org/anthology/D14-1181/](https://www.aclweb.org/anthology/D14-1181/)
*   Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*. 2013. [https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

## Datasets

*   IMDB Dataset of 50K Movie Reviews. Kaggle. [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
*   Rotten Tomatoes Movies and Critic Reviews Dataset. Kaggle. [https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

## Scraping Targets

*   Reddit Discussion Thread for "Paddington in Peru." Reddit. [https://www.reddit.com/r/movies/comments/1ip0uph/official_discussion_paddington_in_peru_spoilers](https://www.reddit.com/r/movies/comments/1ip0uph/official_discussion_paddington_in_peru_spoilers)
*   Rotten Tomatoes Reviews for "Paddington in Peru." Rotten Tomatoes. [https://www.rottentomatoes.com/m/paddington_in_peru/reviews](https://www.rottentomatoes.com/m/paddington_in_peru/reviews)
*   IMDB Reviews for "Paddington in Peru." IMDB. [https://www.imdb.com/title/tt5822536/reviews/?ref_=ttrt_sa_3](https://www.imdb.com/title/tt5822536/reviews/?ref_=ttrt_sa_3)
*   BBC Review for "Paddington in Peru." BBC. [https://www.bbc.com/culture/article/20241103-paddington-in-peru-review-paddington-2](https://www.bbc.com/culture/article/20241103-paddington-in-peru-review-paddington-2)
