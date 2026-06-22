# utils.py
def vectorize_text_tfidf(df, precomputed_vectorizer=None, max_features=5000, min_df=3, max_df=0.85):
    """
    Vectorizes the 'clean_text_transf' column of a DataFrame using TF-IDF.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        precomputed_vectorizer: Optional pre-fitted vectorizer for consistent feature space
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    
    Returns:
        tuple: (tfidf_matrix, tfidf_vectorizer, selected_data)
    """
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    
    # Select columns and drop NaN
    selected_data = df[['sample_id', 'original_id', 'clean_text_transf', 'label', 'method']].copy()
    selected_data = selected_data.dropna(subset=['clean_text_transf'])
    
    # Ensure all values are strings
    selected_data['clean_text_transf'] = selected_data['clean_text_transf'].astype(str)
    
    if precomputed_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=max_features, 
            min_df=min_df, 
            max_df=max_df
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(selected_data['clean_text_transf'])
    else:
        tfidf_vectorizer = precomputed_vectorizer
        tfidf_matrix = tfidf_vectorizer.transform(selected_data['clean_text_transf'])
    
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    return tfidf_matrix, tfidf_vectorizer, selected_data