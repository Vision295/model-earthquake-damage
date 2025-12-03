import pandas as pd
import numpy as np
import json
import warnings
from sklearn.tree import DecisionTreeClassifier

MAX_BINS = 50
MIN_SAMPLES_LEAF = 100

# Suppress sklearn deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

def load_and_merge_data(train_values_path, train_labels_path):
    """Load training data and labels, merge them"""
    df_values = pd.read_csv(train_values_path)
    df_labels = pd.read_csv(train_labels_path)
    df = df_values.merge(df_labels, on='building_id', how='inner')
    return df

def decision_tree_binning(df, feature_name, prediction_col='damage_grade', 
                         max_bins=MAX_BINS, min_samples_leaf=MIN_SAMPLES_LEAF):
    """
    Use Decision Tree to find optimal split points for binning
    More robust than optbinning and works for all types of features
    
    Parameters:
    - df: DataFrame with the feature and target
    - feature_name: name of the numerical feature to bin
    - prediction_col: target column name
    - max_bins: maximum number of bins (max_depth of tree + 1)
    - min_samples_leaf: minimum samples per leaf
    
    Returns:
    - split_points: list of split thresholds
    """
    x = df[feature_name].values.reshape(-1, 1)
    y = df[prediction_col].values
    
    # Use Decision Tree to find splits
    tree = DecisionTreeClassifier(
        max_leaf_nodes=MAX_BINS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42
    )
    tree.fit(x, y)
    
    # Extract thresholds from the tree
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]  # -2 means leaf node
    split_points = sorted(np.unique(thresholds))
    
    return split_points

def apply_binning_to_value(value, splits):
    """
    Apply binning to a single value based on split points
    
    Parameters:
    - value: numerical value to bin
    - splits: list of split points
    
    Returns:
    - bin_label: string representation of the bin interval
    """
    if pd.isna(value):
        return np.nan
    
    if len(splits) == 0:
        return "all"
    
    splits = sorted(splits)
    
    # Find which bin the value falls into
    if value <= splits[0]:
        return f"(-inf, {splits[0]}]"
    
    for i in range(len(splits) - 1):
        if splits[i] < value <= splits[i + 1]:
            return f"({splits[i]}, {splits[i + 1]}]"
    
    return f"({splits[-1]}, inf)"

def create_binning_pipeline(df, prediction_col='damage_grade',
                           features_to_discretize=None,
                           max_bins=MAX_BINS,
                           min_samples_leaf=MIN_SAMPLES_LEAF):
    """
    Create binning transformers for all numerical features using Decision Tree
    
    Returns:
    - binning_pipeline: dict of {feature_name: split_points}
    - df_transformed: DataFrame with binned features
    """
    if features_to_discretize is None:
        features_to_discretize = [
            'age', 
            'area_percentage', 
            'height_percentage', 
            'geo_level_1_id', 
            'geo_level_2_id', 
            'geo_level_3_id'
        ]
    
    df_transformed = df.copy()
    binning_pipeline = {}
    
    print(f"Creating binning pipeline with {len(features_to_discretize)} features...")
    print(f"Features to discretize: {features_to_discretize}")
    print(f"Method: Decision Tree (max_bins={max_bins}, min_samples_leaf={min_samples_leaf})\n")
    
    for feature in features_to_discretize:
        if feature not in df.columns:
            print(f"Warning: {feature} not found in DataFrame, skipping...")
            continue
        
        # Check data type and uniqueness
        n_unique = df[feature].nunique()
        print(f"Processing {feature} (unique values: {n_unique})...")
        
        try:
            # Get split points using decision tree
            split_points = decision_tree_binning(
                df, feature, prediction_col, 
                max_bins, min_samples_leaf
            )
            
            # Store split points
            binning_pipeline[feature] = split_points
            
            # Apply binning
            binned_values = df[feature].apply(
                lambda x: apply_binning_to_value(x, split_points)
            )
            df_transformed[feature] = binned_values.astype('category')
            
            n_bins = len(split_points) + 1
            
            print(f"  ✓ Number of bins: {n_bins}")
            print(f"  ✓ Split points: {split_points}")
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            print(f"  Keeping original column for {feature}")
            print()
    
    print(f"\nBinning pipeline created successfully!")
    print(f"Total features binned: {len(binning_pipeline)}")
    return binning_pipeline, df_transformed

def save_binning_pipeline(binning_pipeline, filename='binning_pipeline.json'):
    """
    Save binning pipeline to JSON file
    """
    pipeline_data = {}
    
    for feature_name, split_points in binning_pipeline.items():
        pipeline_data[feature_name] = {
            'splits': split_points if isinstance(split_points, list) else split_points.tolist(),
            'n_bins': len(split_points) + 1
        }
    
    with open(filename, 'w') as f:
        json.dump(pipeline_data, f, indent=2)
    
    print(f"Binning pipeline saved to: {filename}")
    return filename

def load_binning_pipeline(filename='binning_pipeline.json'):
    """
    Load binning pipeline metadata from JSON file
    
    Returns dict with split points for each feature
    """
    with open(filename, 'r') as f:
        pipeline_data = json.load(f)
    
    print(f"Binning pipeline loaded from: {filename}")
    return pipeline_data

def apply_binning_pipeline(df_new, pipeline_data, 
                          features_to_discretize=None):
    """
    Apply saved binning pipeline to new data
    
    Parameters:
    - df_new: new DataFrame to transform
    - pipeline_data: dict loaded from JSON with split points
    - features_to_discretize: list of features to transform (if None, uses all in pipeline)
    
    Returns:
    - df_transformed: DataFrame with binned features
    """
    if features_to_discretize is None:
        features_to_discretize = list(pipeline_data.keys())
    
    df_transformed = df_new.copy()
    
    print(f"Applying binning pipeline to {len(df_new)} rows...")
    
    for feature in features_to_discretize:
        if feature not in pipeline_data:
            print(f"Warning: {feature} not in pipeline, skipping...")
            continue
        
        if feature not in df_new.columns:
            print(f"Warning: {feature} not in new data, skipping...")
            continue
        
        splits = pipeline_data[feature]['splits']
        
        # Apply binning to each value
        binned_values = df_new[feature].apply(
            lambda x: apply_binning_to_value(x, splits)
        )
        
        df_transformed[feature] = binned_values.astype('category')
        
        print(f"  ✓ Binned {feature}")
    
    print("Binning applied successfully!")
    return df_transformed

def full_training_pipeline(train_values_path, train_labels_path,
                          output_csv='cat_preprocessed_train.csv',
                          output_pipeline='binning_pipeline.json',
                          max_bins=MAX_BINS,
                          min_samples_leaf=MIN_SAMPLES_LEAF):
    """
    Complete pipeline for training: load, bin, and save
    
    Returns:
    - df_transformed: transformed training data
    - binning_pipeline: the split points for each feature
    """
    # Load and merge data
    print("Loading training data...")
    df = load_and_merge_data(train_values_path, train_labels_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns\n")
    
    # Define features to discretize
    features_to_discretize = [
        'age', 
        'area_percentage', 
        'height_percentage', 
        'geo_level_1_id', 
        'geo_level_2_id', 
        'geo_level_3_id'
    ]

    # Create binning pipeline
    binning_pipeline, df_transformed = create_binning_pipeline(
        df,
        prediction_col='damage_grade',
        features_to_discretize=features_to_discretize,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf
    )
    
    # Remove damage_grade column before saving (we only want features, not labels)
    df_transformed = df_transformed.drop('damage_grade', axis=1)
    
    # Save transformed data
    df_transformed.to_csv(output_csv, index=False)
    print(f"\nTransformed data saved to: {output_csv}")
    print(f"Columns in output: {list(df_transformed.columns)}")
    
    # Save pipeline
    save_binning_pipeline(binning_pipeline, output_pipeline)
    
    return df_transformed, binning_pipeline

def bin_single_row(row, pipeline_data, features_to_discretize=None):
    """
    Apply binning to a single row (dict or Series)
    
    Parameters:
    - row: dict or pandas Series with feature values
    - pipeline_data: dict loaded from JSON with split points
    - features_to_discretize: list of features to bin (if None, uses all in pipeline)
    
    Returns:
    - binned_row: dict with binned values
    """
    if features_to_discretize is None:
        features_to_discretize = list(pipeline_data.keys())
    
    # Convert to dict if it's a Series
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    else:
        row_dict = row.copy()
    
    binned_row = row_dict.copy()
    
    for feature in features_to_discretize:
        if feature not in pipeline_data:
            continue
        
        if feature not in row_dict:
            binned_row[feature] = np.nan
            continue
        
        value = row_dict[feature]
        splits = pipeline_data[feature]['splits']
        
        # Apply binning
        binned_value = apply_binning_to_value(value, splits)
        binned_row[feature] = binned_value
    
    return binned_row

def get_bins_for_row(row, pipeline_json='binning_pipeline.json', 
                     features_to_discretize=None):
    """
    Convenience function: Load pipeline and bin a single row
    
    Parameters:
    - row: dict or pandas Series with feature values
    - pipeline_json: path to saved pipeline JSON file
    - features_to_discretize: list of features to bin (if None, uses all in pipeline)
    
    Returns:
    - binned_row: dict with binned values
    
    Example usage:
    >>> row = {'age': 25, 'area_percentage': 50, 'height_percentage': 75, 
    ...        'geo_level_1_id': 10, 'building_id': 12345}
    >>> binned = get_bins_for_row(row, 'binning_pipeline.json')
    >>> print(binned)
    """
    # Load pipeline
    pipeline_data = load_binning_pipeline(pipeline_json)
    
    # Apply binning
    binned_row = bin_single_row(row, pipeline_data, features_to_discretize)
    
    return binned_row

def apply_to_test_data(test_values_path, 
                      pipeline_json='binning_pipeline.json',
                      output_csv='cat_test_values.csv'):
    """
    Apply saved binning pipeline to test data
    
    Parameters:
    - test_values_path: path to test CSV file (e.g., 'test_values.csv')
    - pipeline_json: path to saved binning pipeline JSON
    - output_csv: path for output file (default: 'cat_test_values.csv')
    
    Returns:
    - df_transformed: transformed test data with categorical features
    """
    # Load test data
    print("="*80)
    print("Applying Binning Pipeline to Test Data")
    print("="*80)
    print(f"\nLoading test data from: {test_values_path}")
    df_test = pd.read_csv(test_values_path)
    print(f"✓ Loaded {len(df_test)} samples with {len(df_test.columns)} columns")
    
    # Load pipeline
    print(f"\nLoading binning pipeline from: {pipeline_json}")
    pipeline_data = load_binning_pipeline(pipeline_json)
    print(f"✓ Pipeline loaded with {len(pipeline_data)} features to bin")
    
    # Apply binning to numerical features
    print("\nApplying binning transformations...")
    df_transformed = apply_binning_pipeline(df_test, pipeline_data)
    
    # Save transformed data
    print(f"\nSaving transformed data to: {output_csv}")
    df_transformed.to_csv(output_csv, index=False)
    print(f"✓ Successfully saved {len(df_transformed)} rows")
    
    # Print summary
    print("\n" + "="*80)
    print("TRANSFORMATION SUMMARY")
    print("="*80)
    print(f"Input file:  {test_values_path}")
    print(f"Output file: {output_csv}")
    print(f"Rows:        {len(df_transformed)}")
    print(f"Columns:     {len(df_transformed.columns)}")
    
    # Show which columns are now categorical
    categorical_cols = df_transformed.select_dtypes(include=['category']).columns.tolist()
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        n_categories = df_transformed[col].nunique()
        print(f"  - {col}: {n_categories} categories")
    
    print("="*80)
    
    return df_transformed


if __name__ == "__main__":
    print("="*80)
    print("Supervised Binning for CatBoost - Training Pipeline")
    print("="*80)
    print("\nUsing Decision Tree-based binning (no external dependencies)\n")
    
    # TRAINING PHASE
    print("\n" + "="*80)
    print("PHASE 1: TRAINING - Create binning pipeline")
    print("="*80 + "\n")
    
    df_train, pipeline = full_training_pipeline(
        train_values_path='train_values.csv',
        train_labels_path='train_labels.csv',
        output_csv='cat_preprocessed_train.csv',
        output_pipeline='binning_pipeline.json',
        max_bins=MAX_BINS,  # Maximum number of bins per feature
        min_samples_leaf=MIN_SAMPLES_LEAF  # Minimum samples per bin
    )
    
    # TEST PHASE
    print("\n" + "="*80)
    print("PHASE 2: TESTING - Apply binning to new data")
    print("="*80 + "\n")
    
    # Apply to test data
    df_test = apply_to_test_data(
        test_values_path='test_values.csv',
        pipeline_json='binning_pipeline.json',
        output_csv='cat_test_values.csv'
    )
    
    # SINGLE ROW EXAMPLE
    print("\n" + "="*80)
    print("PHASE 3: SINGLE ROW - Example of binning a single row")
    print("="*80 + "\n")
    
    # Example: bin a single row
    example_row = {
        'building_id': 99999,
        'geo_level_1_id': 10,
        'geo_level_2_id': 500,
        'geo_level_3_id': 5000,
        'age': 25,
        'area_percentage': 50,
        'height_percentage': 75
    }
    
    print("Original row:")
    for k, v in example_row.items():
        print(f"  {k}: {v}")
    
    binned_row = get_bins_for_row(example_row, 'binning_pipeline.json')
    
    print("\nBinned row:")
    for k, v in binned_row.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*80)
    print("COMPLETE! Files created:")
    print("  - cat_preprocessed_train.csv (transformed training data)")
    print("  - cat_test_values.csv (transformed test data)")
    print("  - binning_pipeline.json (saved bin splits)")
    print("\nTo transform new data:")
    print("  1. Full dataset: apply_to_test_data('test_values.csv', 'binning_pipeline.json', 'cat_test_values.csv')")
    print("  2. Single row: get_bins_for_row(row_dict, 'binning_pipeline.json')")
    print("="*80)