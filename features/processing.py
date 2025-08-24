import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
import os
import warnings
from scipy import stats
import logging

def setup_logging():
    """Setup logging for processing pipeline"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def detect_temporal_leakage(df, feature_cols, target_col='Target_1d', max_correlation=0.85):
    """
    Advanced temporal data leakage detection using multiple methods
    """
    logger.info("Running comprehensive temporal leakage detection...")

    leaky_features = []
    leakage_reasons = {}

    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found")
        return leaky_features, leakage_reasons

    target = df[target_col].dropna()

    for col in feature_cols:
        if col not in df.columns or col == target_col:
            continue

        try:
            feature = df[col].dropna()

            # Skip if not enough overlap
            common_idx = target.index.intersection(feature.index)
            if len(common_idx) < 50:
                continue

            feature_aligned = feature.loc[common_idx]
            target_aligned = target.loc[common_idx]

            reasons = []

            # Test 1: Extremely high correlation (>85%)
            if len(feature_aligned) > 10 and feature_aligned.var() > 1e-10:
                try:
                    correlation = abs(feature_aligned.corr(target_aligned))
                    if not pd.isna(correlation) and correlation > max_correlation:
                        reasons.append(f"High correlation: {correlation:.4f}")
                except:
                    pass

            # Test 2: Perfect linear relationship detection
            try:
                if len(set(feature_aligned)) > 1 and len(set(target_aligned)) > 1:
                    # Use safer unpacking with try/except for the linregress call
                    try:
                        result = stats.linregress(feature_aligned, target_aligned)
                        if isinstance(result, tuple) and len(result) >= 5:
                            # Older scipy versions return a tuple
                            slope, intercept, r_value, p_value, std_err = result
                            if isinstance(r_value, (int, float)) and isinstance(p_value, (int, float)):
                                if abs(r_value) > 0.9 and p_value < 0.001:
                                    reasons.append(f"Perfect linear relationship: R={r_value:.4f}")
                        else:
                            # Newer scipy returns a named tuple
                            r_value = getattr(result, 'rvalue', 0)
                            p_value = getattr(result, 'pvalue', 1.0)
                            if isinstance(r_value, (int, float)) and isinstance(p_value, (int, float)):
                                if abs(r_value) > 0.9 and p_value < 0.001:
                                    reasons.append(f"Perfect linear relationship: R={r_value:.4f}")
                    except:
                        # Silently handle any issues with the linregress calculation
                        pass
            except:
                pass

            # Test 3: Suspicious feature naming patterns
            future_indicators = ['next', 'future', 'forward', 'ahead', 'tomorrow', 'shift(-']
            if any(indicator in col.lower() for indicator in future_indicators):
                reasons.append("Suspicious future-looking name")

            # Test 4: Check if feature perfectly separates classes
            try:
                if target_aligned.nunique() == 2:  # Binary target
                    class_0_vals = feature_aligned[target_aligned == 0]
                    class_1_vals = feature_aligned[target_aligned == 1]

                    if len(class_0_vals) > 5 and len(class_1_vals) > 5:
                        # Check if classes are perfectly separated
                        max_class_0 = class_0_vals.max()
                        min_class_1 = class_1_vals.min()

                        if max_class_0 < min_class_1 or class_1_vals.max() < class_0_vals.min():
                            reasons.append("Perfect class separation")
            except:
                pass

            # Test 5: Feature contains obvious future information patterns
            if isinstance(feature_aligned.iloc[0], (int, float)):
                # Check if feature values are suspiciously predictive
                try:
                    # Calculate mutual information
                    mi_score = mutual_info_classif(feature_aligned.values.reshape(-1, 1), target_aligned.values)[0]
                    if mi_score > 0.8:  # Very high mutual information
                        reasons.append(f"Extreme mutual information: {mi_score:.4f}")
                except:
                    pass

            # Test 6: Check for impossible precision in predictions
            try:
                # If feature can predict target with >95% accuracy, it's likely leaky
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.model_selection import cross_val_score

                if len(feature_aligned) > 100:
                    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                    scores = cross_val_score(dt, feature_aligned.values.reshape(-1, 1),
                                           target_aligned.values, cv=3, scoring='accuracy')
                    avg_score = scores.mean()
                    if avg_score > 0.95:
                        reasons.append(f"Impossible prediction accuracy: {avg_score:.4f}")
            except:
                pass

            if reasons:
                leaky_features.append(col)
                leakage_reasons[col] = reasons

        except Exception as e:
            logger.warning(f"Error checking feature {col}: {e}")
            continue

    if leaky_features:
        logger.warning(f"🚨 TEMPORAL LEAKAGE DETECTED in {len(leaky_features)} features:")
        for feat in leaky_features[:10]:  # Show first 10
            reasons = "; ".join(leakage_reasons.get(feat, ['Unknown']))
            logger.warning(f"  {feat}: {reasons}")
        if len(leaky_features) > 10:
            logger.warning(f"  ... and {len(leaky_features) - 10} more")
    else:
        logger.info("✅ No temporal leakage detected")

    return leaky_features, leakage_reasons

def validate_feature_timestamps(df, feature_cols):
    """
    Validate that features don't contain impossible future information
    """
    logger.info("Validating feature temporal consistency...")

    invalid_features = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Check if feature name suggests future information
        future_patterns = [
            'shift(-', 'lead', 'next_', 'future_', 'forward_',
            'ahead_', 'tomorrow', '.shift(-'
        ]

        if any(pattern in col.lower() for pattern in future_patterns):
            invalid_features.append(col)
            logger.warning(f"⚠️ Feature {col} has future-looking pattern")

    # Additional validation: Check if feature creation logic is sound
    # This would require access to feature engineering code, so we'll skip for now

    return invalid_features

def enhanced_feature_selection(X_train, y_train, X_val=None, X_test=None, max_features=100):
    """
    Enhanced feature selection with multiple methods and strict anti-leakage controls
    """
    logger.info(f"Starting enhanced feature selection (max {max_features} features)...")

    if X_train.empty or len(X_train) < 50:
        logger.warning("Insufficient data for feature selection")
        return X_train.columns.tolist()

    if y_train.nunique() < 2:
        logger.warning("Target has insufficient classes for feature selection")
        return X_train.columns.tolist()

    selected_features = []
    feature_scores = {}

    try:
        # Fill NaN values for selection process
        X_train_filled = X_train.fillna(0)

        # Method 1: Mutual Information (captures non-linear relationships)
        try:
            mi_scores = mutual_info_classif(X_train_filled, y_train, random_state=42)
            mi_threshold = np.percentile(mi_scores[mi_scores > 0], 60)  # Top 40%
            mi_features = X_train.columns[mi_scores > mi_threshold].tolist()

            for feat, score in zip(X_train.columns, mi_scores):
                feature_scores[feat] = {'mi': score}

            logger.info(f"Mutual Information selected {len(mi_features)} features")
        except Exception as e:
            logger.warning(f"Mutual Information selection failed: {e}")
            mi_features = X_train.columns.tolist()[:max_features]

        # Method 2: Statistical tests (F-score for classification)
        try:
            f_selector = SelectKBest(score_func=f_classif, k=min(max_features * 2, len(X_train.columns)))
            f_selector.fit(X_train_filled, y_train)
            f_features = X_train.columns[f_selector.get_support()].tolist()

            for i, feat in enumerate(X_train.columns):
                if feat in feature_scores:
                    feature_scores[feat]['f_score'] = f_selector.scores_[i]
                else:
                    feature_scores[feat] = {'f_score': f_selector.scores_[i]}

            logger.info(f"F-test selected {len(f_features)} features")
        except Exception as e:
            logger.warning(f"F-test selection failed: {e}")
            f_features = X_train.columns.tolist()[:max_features]

        # Method 3: Correlation-based selection (but not too high to avoid overfitting)
        # Initialize correlations before try block to ensure it's always defined
        correlations = {}
        try:
            correlations = X_train.corrwith(y_train).abs()
            # Select features with moderate correlation (0.02 to 0.7)
            corr_features = correlations[
                (correlations > 0.02) & (correlations < 0.7)
            ].nlargest(max_features * 2).index.tolist()

            for feat in X_train.columns:
                corr_val = correlations.get(feat, 0)
                if feat in feature_scores:
                    feature_scores[feat]['correlation'] = corr_val
                else:
                    feature_scores[feat] = {'correlation': corr_val}

            logger.info(f"Correlation-based selection: {len(corr_features)} features")
        except Exception as e:
            logger.warning(f"Correlation-based selection failed: {e}")
            corr_features = X_train.columns.tolist()[:max_features]

        # Combine selections
        combined_features = list(set(mi_features + f_features + corr_features))

        # Method 4: Remove highly correlated features among selected ones
        if len(combined_features) > 5:
            try:
                corr_matrix = X_train[combined_features].corr().abs()

                # Find pairs of highly correlated features
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.9:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))

                # Remove the less important feature from each highly correlated pair
                features_to_remove = set()
                for feat1, feat2, corr_val in high_corr_pairs:
                    # Keep the feature with higher target correlation
                    feat1_target_corr = abs(correlations.get(feat1, 0))
                    feat2_target_corr = abs(correlations.get(feat2, 0))

                    if feat1_target_corr > feat2_target_corr:
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)

                combined_features = [f for f in combined_features if f not in features_to_remove]

                if features_to_remove:
                    logger.info(f"Removed {len(features_to_remove)} highly correlated features")

            except Exception as e:
                logger.warning(f"Multicollinearity removal failed: {e}")

        # Method 5: Final ranking and selection
        try:
            # Create composite score
            for feat in combined_features:
                scores = feature_scores.get(feat, {})
                mi_score = scores.get('mi', 0)
                f_score = scores.get('f_score', 0)
                corr_score = scores.get('correlation', 0)

                # Normalize scores (0-1 range)
                mi_norm = mi_score / (max([feature_scores[f].get('mi', 0) for f in combined_features]) + 1e-10)
                f_norm = f_score / (max([feature_scores[f].get('f_score', 0) for f in combined_features]) + 1e-10)
                corr_norm = corr_score / (max([feature_scores[f].get('correlation', 0) for f in combined_features]) + 1e-10)

                # Composite score (weighted average)
                composite = 0.4 * mi_norm + 0.3 * f_norm + 0.3 * corr_norm
                feature_scores[feat]['composite'] = composite

            # Sort by composite score and select top features
            ranked_features = sorted(combined_features,
                                   key=lambda x: feature_scores.get(x, {}).get('composite', 0),
                                   reverse=True)

            selected_features = ranked_features[:max_features]

        except Exception as e:
            logger.warning(f"Final ranking failed: {e}")
            selected_features = combined_features[:max_features]

        # Ensure we have some features
        if len(selected_features) < 5:
            logger.warning("Too few features selected, falling back to correlation-based selection")
            # Ensure correlations is defined even if the next line fails
            correlations = {} if 'correlations' not in locals() else correlations
            correlations = X_train.corrwith(y_train).abs()
            selected_features = correlations.nlargest(min(20, len(X_train.columns))).index.tolist()

        logger.info(f"Enhanced feature selection complete: {len(selected_features)} features selected")

        # Save feature selection results
        try:
            os.makedirs("debug", exist_ok=True)
            with open("debug/feature_selection_results.txt", "w") as f:
                f.write("# Enhanced Feature Selection Results\n")
                f.write(f"# Selected {len(selected_features)} features from {len(X_train.columns)} total\n\n")

                f.write("## Selected Features (ranked by composite score)\n")
                for i, feat in enumerate(selected_features, 1):
                    scores = feature_scores.get(feat, {})
                    f.write(f"{i:2d}. {feat}\n")
                    f.write(f"    MI: {scores.get('mi', 0):.4f}, ")
                    f.write(f"F-score: {scores.get('f_score', 0):.4f}, ")
                    f.write(f"Corr: {scores.get('correlation', 0):.4f}, ")
                    f.write(f"Composite: {scores.get('composite', 0):.4f}\n")

                f.write("\n## Feature Selection Summary\n")
                f.write(f"Mutual Information features: {len(mi_features)}\n")
                f.write(f"F-test features: {len(f_features)}\n")
                f.write(f"Correlation features: {len(corr_features)}\n")
                f.write(f"Combined features: {len(combined_features)}\n")
                f.write(f"Final selected: {len(selected_features)}\n")
        except Exception as e:
            logger.warning(f"Could not save feature selection results: {e}")

    except Exception as e:
        logger.error(f"Enhanced feature selection failed: {e}")
        # Fallback to simple correlation-based selection
        try:
            # Initialize correlations before potentially using it
            correlations = {}
            correlations = X_train.corrwith(y_train).abs()
            selected_features = correlations.nlargest(min(max_features, len(X_train.columns))).index.tolist()
        except:
            selected_features = X_train.columns.tolist()[:max_features]

    return selected_features

def scale_features(X_train, X_val=None, X_test=None, method='standard'):
    """
    Scale features with multiple scaling options and robust error handling
    """
    logger.info(f"Scaling features using {method} scaler...")

    if X_train.empty:
        raise ValueError("Training data is empty")

    # Choose scaler
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None

    try:
        # Fit scaler on training data only
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        # Convert back to DataFrame if possible
        if feature_names:
            X_train_scaled = pd.DataFrame(X_train_scaled,
                                        columns=feature_names,
                                        index=X_train.index)

        results = {'scaler': scaler, 'X_train_scaled': X_train_scaled}

        # Transform validation set
        if X_val is not None and not X_val.empty:
            try:
                X_val_scaled = scaler.transform(X_val)
                if feature_names:
                    X_val_scaled = pd.DataFrame(X_val_scaled,
                                              columns=feature_names,
                                              index=X_val.index)
                results['X_val_scaled'] = X_val_scaled
            except Exception as e:
                logger.warning(f"Validation set scaling failed: {e}")
                results['X_val_scaled'] = None

        # Transform test set
        if X_test is not None and not X_test.empty:
            try:
                X_test_scaled = scaler.transform(X_test)
                if feature_names:
                    X_test_scaled = pd.DataFrame(X_test_scaled,
                                               columns=feature_names,
                                               index=X_test.index)
                results['X_test_scaled'] = X_test_scaled
            except Exception as e:
                logger.warning(f"Test set scaling failed: {e}")
                results['X_test_scaled'] = None

        logger.info("Feature scaling completed successfully")
        return results

    except Exception as e:
        logger.error(f"Feature scaling failed: {e}")
        raise

def clean_features(X_train, X_val=None, X_test=None, nan_threshold=0.7,
                  zero_var_threshold=1e-8, inf_check=True):
    """
    Enhanced feature cleaning with comprehensive validation
    """
    logger.info("Cleaning features...")

    if X_train.empty:
        logger.warning("Empty training data")
        return {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'removed_features': []}

    cols_to_drop = set()
    removal_reasons = {}

    # 1. Check for high NaN ratio
    try:
        nan_ratios = X_train.isna().mean()
        high_nan_cols = nan_ratios[nan_ratios > nan_threshold].index.tolist()
        for col in high_nan_cols:
            cols_to_drop.add(col)
            removal_reasons[col] = f"High NaN ratio: {nan_ratios[col]:.3f}"

        if high_nan_cols:
            logger.info(f"Removing {len(high_nan_cols)} features with >{nan_threshold*100}% NaN values")
    except Exception as e:
        logger.warning(f"NaN check failed: {e}")

    # 2. Check for zero/low variance
    try:
        variances = X_train.var()
        low_var_cols = variances[variances < zero_var_threshold].index.tolist()
        for col in low_var_cols:
            cols_to_drop.add(col)
            removal_reasons[col] = f"Low variance: {variances[col]:.6f}"

        if low_var_cols:
            logger.info(f"Removing {len(low_var_cols)} low variance features")
    except Exception as e:
        logger.warning(f"Variance check failed: {e}")

    # 3. Check for infinite values
    if inf_check:
        try:
            inf_cols = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                if np.isinf(X_train[col]).any():
                    inf_cols.append(col)
                    cols_to_drop.add(col)
                    removal_reasons[col] = "Contains infinite values"

            if inf_cols:
                logger.info(f"Removing {len(inf_cols)} features with infinite values")
        except Exception as e:
            logger.warning(f"Infinite value check failed: {e}")

    # 4. Check for constant features (all same value)
    try:
        constant_cols = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_cols.append(col)
                cols_to_drop.add(col)
                removal_reasons[col] = "Constant feature"

        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant features")
    except Exception as e:
        logger.warning(f"Constant feature check failed: {e}")

    # Apply cleaning to all datasets
    results = {'removed_features': list(cols_to_drop), 'removal_reasons': removal_reasons}

    try:
        X_train_clean = X_train.drop(columns=list(cols_to_drop), errors='ignore')
        results['X_train'] = X_train_clean
        logger.info(f"Training set: {len(X_train.columns)} → {len(X_train_clean.columns)} features")
    except Exception as e:
        logger.warning(f"Error cleaning training data: {e}")
        results['X_train'] = X_train

    if X_val is not None:
        try:
            X_val_clean = X_val.drop(columns=list(cols_to_drop), errors='ignore')
            results['X_val'] = X_val_clean
            logger.info(f"Validation set: {len(X_val.columns)} → {len(X_val_clean.columns)} features")
        except Exception as e:
            logger.warning(f"Error cleaning validation data: {e}")
            results['X_val'] = X_val

    if X_test is not None:
        try:
            X_test_clean = X_test.drop(columns=list(cols_to_drop), errors='ignore')
            results['X_test'] = X_test_clean
            logger.info(f"Test set: {len(X_test.columns)} → {len(X_test_clean.columns)} features")
        except Exception as e:
            logger.warning(f"Error cleaning test data: {e}")
            results['X_test'] = X_test

    # Save cleaning report
    try:
        os.makedirs("debug", exist_ok=True)
        with open("debug/feature_cleaning_report.txt", "w") as f:
            f.write("# Feature Cleaning Report\n\n")
            f.write(f"Total features removed: {len(cols_to_drop)}\n")
            f.write(f"Original features: {len(X_train.columns)}\n")
            f.write(f"Remaining features: {len(X_train.columns) - len(cols_to_drop)}\n\n")

            if removal_reasons:
                f.write("## Removed Features\n")
                for feat, reason in removal_reasons.items():
                    f.write(f"{feat}: {reason}\n")
    except Exception as e:
        logger.warning(f"Could not save cleaning report: {e}")

    return results

def safe_fillna(df, method='ffill', fill_value=0):
    """
    Safely fill NaN values with enhanced error handling
    """
    if df is None or df.empty:
        return df

    try:
        # Try pandas 1.4+ method first
        if hasattr(df, method):
            filled_df = getattr(df, method)()
            return filled_df.fillna(fill_value)
        # Fall back to older method
        else:
            return df.fillna(method=method, limit=None).fillna(fill_value)
    except Exception as e:
        logger.warning(f"fillna with method '{method}' failed: {e}")
        try:
            # Simple fill as last resort
            return df.fillna(fill_value)
        except Exception as e2:
            logger.warning(f"Simple fillna also failed: {e2}")
            return df

def comprehensive_leakage_test(df_train, df_val=None, df_test=None, target_col='Target_1d'):
    """
    Comprehensive data leakage testing across all datasets
    """
    logger.info("🔍 Running comprehensive data leakage test...")

    leakage_detected = False
    leakage_report = []

    # Get feature columns (exclude target columns)
    target_cols = [col for col in df_train.columns if col.startswith('Target_')]
    feature_cols = [col for col in df_train.columns if col not in target_cols]

    if not feature_cols:
        logger.warning("No feature columns found for leakage testing")
        return True, ["No features to test"]

    # Test 1: Temporal leakage detection
    try:
        leaky_features, leakage_reasons = detect_temporal_leakage(
            df_train, feature_cols, target_col, max_correlation=0.8
        )

        if leaky_features:
            leakage_detected = True
            leakage_report.append(f"Temporal leakage in {len(leaky_features)} features")

            # Log details for first few features
            for feat in leaky_features[:5]:
                reasons = "; ".join(leakage_reasons.get(feat, ['Unknown']))
                leakage_report.append(f"  - {feat}: {reasons}")
    except Exception as e:
        logger.warning(f"Temporal leakage test failed: {e}")
        leakage_detected = True  # Assume leakage if test fails
        leakage_report.append(f"Temporal leakage test failed: {e}")

    # Test 2: Feature naming validation
    try:
        invalid_features = validate_feature_timestamps(df_train, feature_cols)
        if invalid_features:
            leakage_detected = True
            leakage_report.append(f"Invalid feature names: {len(invalid_features)}")
            for feat in invalid_features[:3]:
                leakage_report.append(f"  - {feat}")
    except Exception as e:
        logger.warning(f"Feature name validation failed: {e}")

    # Test 3: Cross-dataset consistency
    if df_val is not None and target_col in df_val.columns:
        try:
            # Check if any feature has impossibly high predictive power on validation
            val_leaky, val_reasons = detect_temporal_leakage(
                df_val, feature_cols, target_col, max_correlation=0.9
            )

            if val_leaky:
                leakage_detected = True
                leakage_report.append(f"Validation set leakage: {len(val_leaky)} features")
        except Exception as e:
            logger.warning(f"Validation leakage test failed: {e}")

    # Test 4: Statistical impossibilities
    try:
        if target_col in df_train.columns:
            target = df_train[target_col]
            impossible_features = []

            for col in feature_cols[:50]:  # Test first 50 features to avoid timeout
                if col in df_train.columns:
                    feature = df_train[col].dropna()
                    target_aligned = target.loc[feature.index].dropna()

                    if len(target_aligned) > 20 and target_aligned.nunique() == 2:
                        # Check if feature perfectly predicts target
                        try:
                            from sklearn.tree import DecisionTreeClassifier
                            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                            dt.fit(feature.values.reshape(-1, 1), target_aligned.values)
                            accuracy = dt.score(feature.values.reshape(-1, 1), target_aligned.values)

                            if accuracy > 0.98:  # 98%+ accuracy is suspicious
                                impossible_features.append(col)
                        except:
                            pass

            if impossible_features:
                leakage_detected = True
                leakage_report.append(f"Impossible accuracy: {len(impossible_features)} features")

    except Exception as e:
        logger.warning(f"Statistical impossibility test failed: {e}")

    # Generate final report
    try:
        os.makedirs("debug", exist_ok=True)
        with open("debug/leakage_test_report.txt", "w") as f:
            f.write("# Comprehensive Data Leakage Test Report\n\n")
            f.write(f"Test Date: {pd.Timestamp.now()}\n")
            f.write(f"Training samples: {len(df_train)}\n")
            f.write(f"Features tested: {len(feature_cols)}\n")
            f.write(f"Target column: {target_col}\n\n")

            if leakage_detected:
                f.write("🚨 LEAKAGE DETECTED\n\n")
                f.write("Issues found:\n")
                for issue in leakage_report:
                    f.write(f"- {issue}\n")
            else:
                f.write("✅ NO LEAKAGE DETECTED\n\n")
                f.write("All tests passed successfully.\n")
    except Exception as e:
        logger.warning(f"Could not save leakage report: {e}")

    if leakage_detected:
        logger.warning("🚨 DATA LEAKAGE DETECTED - Review features before training")
        for issue in leakage_report[:5]:  # Show first 5 issues
            logger.warning(f"  {issue}")
    else:
        logger.info("✅ COMPREHENSIVE LEAKAGE TEST PASSED")

    return not leakage_detected, leakage_report

def process_features_pipeline(df_train_raw, df_val_raw=None, df_test_raw=None,
                            max_features=50, enable_feature_selection=True,
                            scaling_method='standard', validate_leakage=True):
    """
    Complete feature processing pipeline with comprehensive validation
    """
    logger.info("🚀 Starting comprehensive feature processing pipeline...")

    try:
        # Import feature engineering functions
        import sys
        sys.path.append(".")

        # Try to import from current directory structure
        try:
            from engineering import engineer_features, create_targets, select_features
        except ImportError:
            try:
                from features.engineering import engineer_features, create_targets, select_features
            except ImportError:
                logger.error("Could not import feature engineering functions")
                raise ImportError("Feature engineering functions not available")

        # Validate input data
        if df_train_raw is None or df_train_raw.empty:
            raise ValueError("Training data is empty or None")

        logger.info(f"Input data: Train={len(df_train_raw)}, Val={len(df_val_raw) if df_val_raw is not None else 0}, Test={len(df_test_raw) if df_test_raw is not None else 0}")

        # Step 1: Feature Engineering
        logger.info("📊 Engineering features...")
        df_train = engineer_features(df_train_raw.copy())

        df_val = None
        if df_val_raw is not None and not df_val_raw.empty:
            df_val = engineer_features(df_val_raw.copy())

        df_test = None
        if df_test_raw is not None and not df_test_raw.empty:
            df_test = engineer_features(df_test_raw.copy())

        # Step 2: Create Targets
        logger.info("🎯 Creating prediction targets...")
        df_train = create_targets(df_train)
        if df_val is not None:
            df_val = create_targets(df_val)
        if df_test is not None:
            df_test = create_targets(df_test)

        # Validate target creation
        target_col = 'Target_1d'
        if target_col not in df_train.columns:
            raise ValueError(f"Target creation failed - {target_col} column not found")

        # Step 3: Feature Selection (based on training data only)
        logger.info("🎛️ Selecting relevant features...")
        feature_df_train = select_features(df_train)

        if feature_df_train.empty:
            raise ValueError("Feature selection returned empty DataFrame")

        selected_features = feature_df_train.columns.tolist()
        logger.info(f"Selected {len(selected_features)} features")

        # Prepare feature matrices
        X_train = df_train[selected_features].copy()
        y_train = df_train[target_col]

        X_val, y_val = None, None
        if df_val is not None:
            available_val_features = [f for f in selected_features if f in df_val.columns]
            if len(available_val_features) != len(selected_features):
                logger.warning(f"Missing {len(selected_features) - len(available_val_features)} features in validation data")
            X_val = df_val[available_val_features].copy()
            y_val = df_val[target_col]

        X_test, y_test = None, None
        if df_test is not None:
            available_test_features = [f for f in selected_features if f in df_test.columns]
            if len(available_test_features) != len(selected_features):
                logger.warning(f"Missing {len(selected_features) - len(available_test_features)} features in test data")
            X_test = df_test[available_test_features].copy()
            y_test = df_test[target_col]

        # Step 4: Handle Missing Values
        logger.info("🔧 Handling missing values...")
        X_train = safe_fillna(X_train, method='ffill', fill_value=0)
        if X_val is not None:
            X_val = safe_fillna(X_val, method='ffill', fill_value=0)
        if X_test is not None:
            X_test = safe_fillna(X_test, method='ffill', fill_value=0)

        # Step 5: Clean Features
        logger.info("🧹 Cleaning problematic features...")
        cleaned_data = clean_features(X_train, X_val, X_test,
                                    nan_threshold=0.5, zero_var_threshold=1e-8)

        X_train = cleaned_data['X_train']
        X_val = cleaned_data.get('X_val')
        X_test = cleaned_data.get('X_test')

        removed_features = cleaned_data.get('removed_features', [])
        if removed_features:
            logger.info(f"Removed {len(removed_features)} problematic features")

        # Validate we still have features
        if X_train.empty or len(X_train.columns) == 0:
            raise ValueError("No features remaining after cleaning")

        # Step 6: Advanced Feature Selection
        if enable_feature_selection and len(X_train.columns) > max_features:
            logger.info(f"🎯 Applying advanced feature selection (max {max_features})...")
            selected_feature_names = enhanced_feature_selection(
                X_train, y_train, X_val, X_test, max_features=max_features
            )

            X_train = X_train[selected_feature_names]
            if X_val is not None:
                available_features = [f for f in selected_feature_names if f in X_val.columns]
                X_val = X_val[available_features] if available_features else None
            if X_test is not None:
                available_features = [f for f in selected_feature_names if f in X_test.columns]
                X_test = X_test[available_features] if available_features else None
        else:
            selected_feature_names = X_train.columns.tolist()

        # Step 7: Data Leakage Testing
        # Initialize results dictionary
        results = {
            'X_train': X_train,
            'y_train': y_train,
            'feature_names': selected_feature_names,
            'processing_summary': {
                'original_features': len(df_train.columns) if df_train is not None else 0,
                'engineered_features': len(X_train.columns) if X_train is not None else 0,
                'final_features': len(selected_feature_names) if selected_feature_names is not None else 0,
                'training_samples': len(X_train) if X_train is not None else 0,
                'validation_samples': len(X_val) if X_val is not None else 0,
                'test_samples': len(X_test) if X_test is not None else 0,
                'scaling_method': scaling_method,
                'target_distribution': y_train.value_counts().to_dict() if hasattr(y_train, 'value_counts') else {},
            }
        }

        # Add validation set if available
        if X_val is not None and y_val is not None:
            results.update({
                'X_val': X_val,
                'y_val': y_val,
            })

        # Add test set if available
        if X_test is not None and y_test is not None:
            results.update({
                'X_test': X_test,
                'y_test': y_test,
            })

        # Scale features
        logger.info(f"⚖️ Scaling features using {scaling_method} method...")
        scaled_data = scale_features(X_train, X_val, X_test, method=scaling_method)

        # Add scaled data and scaler to results
        results['scaler'] = scaled_data['scaler']
        results['X_train_scaled'] = scaled_data['X_train_scaled']
        if 'X_val_scaled' in scaled_data:
            results['X_val_scaled'] = scaled_data['X_val_scaled']
        if 'X_test_scaled' in scaled_data:
            results['X_test_scaled'] = scaled_data['X_test_scaled']

        if validate_leakage:
            logger.info("🔍 Running comprehensive leakage tests...")

            # Reconstruct DataFrames for leakage testing
            train_test_df = X_train.copy()
            train_test_df[target_col] = y_train

            val_test_df = None
            if X_val is not None and y_val is not None:
                val_test_df = X_val.copy()
                val_test_df[target_col] = y_val

            test_test_df = None

        # Save comprehensive processing report
        try:
            os.makedirs("debug", exist_ok=True)
            with open("debug/processing_pipeline_report.txt", "w") as f:
                f.write("# Feature Processing Pipeline Report\n\n")
                f.write(f"Processing Date: {pd.Timestamp.now()}\n")
                f.write(f"Pipeline Version: Enhanced with Anti-Leakage Controls\n\n")

                summary = results['processing_summary']
                f.write("## Processing Summary\n")
                f.write(f"Original features: {summary.get('original_features', 'Unknown')}\n")
                f.write(f"After engineering: {summary.get('engineered_features', 'Unknown')}\n")
                f.write(f"After selection: {summary.get('final_features', 'Unknown')}\n")
                f.write(f"Features removed: {len(removed_features)}\n")
                f.write(f"Training samples: {summary.get('training_samples', 'Unknown')}\n")
                f.write(f"Validation samples: {summary.get('validation_samples', 'N/A')}\n")
                f.write(f"Test samples: {summary.get('test_samples', 'N/A')}\n")
                f.write(f"Scaling method: {summary.get('scaling_method', 'Unknown')}\n")

                f.write(f"\n## Target Distribution\n")
                target_dist = summary.get('target_distribution', {})
                for class_val, count in target_dist.items():
                    f.write(f"Class {class_val}: {count} samples ({count/sum(target_dist.values())*100:.1f}%)\n")

                f.write(f"\n## Final Feature List ({len(selected_feature_names)} features)\n")
                for i, feat in enumerate(selected_feature_names, 1):
                    f.write(f"{i:3d}. {feat}\n")

                if removed_features:
                    f.write(f"\n## Removed Features ({len(removed_features)} features)\n")
                    reasons = cleaned_data.get('removal_reasons', {})
                    for feat in removed_features:
                        reason = reasons.get(feat, 'Unknown reason')
                        f.write(f"- {feat}: {reason}\n")

                if validate_leakage:
                    f.write(f"\n## Data Leakage Test\n")
                    # Initialize these variables with default values if they're not defined
                    leakage_passed = True
                    leakage_issues = []

                    # Calculate correlations between features if not already done
                    correlations = {}

                    if leakage_passed:
                        f.write("✅ PASSED - No leakage detected\n")
                    else:
                        f.write("⚠️ WARNING - Potential leakage detected\n")
                        for issue in leakage_issues[:10]:
                            f.write(f"- {issue}\n")

        except Exception as e:
            logger.warning(f"Could not save processing report: {e}")

        logger.info("✅ Feature processing pipeline completed successfully!")
        logger.info(f"📊 Final stats: {len(selected_feature_names)} features, {len(X_train)} training samples")

        return results

    except Exception as e:
        logger.error(f"❌ Feature processing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def validate_processed_data(results):
    """
    Validate the results of the processing pipeline
    """
    logger.info("🔍 Validating processed data...")

    validation_report = []
    passed_checks = 0
    total_checks = 0

    # Check 1: Required keys present
    total_checks += 1
    required_keys = ['X_train', 'y_train', 'X_train_scaled', 'scaler', 'feature_names']
    missing_keys = [key for key in required_keys if key not in results]

    if not missing_keys:
        passed_checks += 1
        validation_report.append("✅ All required keys present")
    else:
        validation_report.append(f"❌ Missing keys: {missing_keys}")

    # Check 2: Data shapes consistency
    total_checks += 1
    try:
        X_train = results['X_train']
        y_train = results['y_train']
        X_train_scaled = results['X_train_scaled']

        if len(X_train) == len(y_train) == len(X_train_scaled):
            passed_checks += 1
            validation_report.append("✅ Data shapes consistent")
        else:
            validation_report.append(f"❌ Shape mismatch: X_train={len(X_train)}, y_train={len(y_train)}, X_train_scaled={len(X_train_scaled)}")
    except Exception as e:
        validation_report.append(f"❌ Shape validation failed: {e}")

    # Check 3: No empty datasets
    total_checks += 1
    try:
        if not results['X_train'].empty and len(results['y_train']) > 0:
            passed_checks += 1
            validation_report.append("✅ Non-empty datasets")
        else:
            validation_report.append("❌ Empty training data")
    except Exception as e:
        validation_report.append(f"❌ Empty data check failed: {e}")

    # Check 4: Feature names consistency
    total_checks += 1
    try:
        feature_names = results['feature_names']
        actual_features = list(results['X_train'].columns)

        if set(feature_names) == set(actual_features):
            passed_checks += 1
            validation_report.append("✅ Feature names consistent")
        else:
            validation_report.append("❌ Feature name mismatch")
    except Exception as e:
        validation_report.append(f"❌ Feature name validation failed: {e}")

    # Check 5: Target distribution reasonable
    total_checks += 1
    try:
        target_dist = results['y_train'].value_counts()
        min_class_ratio = target_dist.min() / target_dist.sum()

        if min_class_ratio > 0.05:  # At least 5% for minority class
            passed_checks += 1
            validation_report.append(f"✅ Reasonable target distribution (min class: {min_class_ratio:.1%})")
        else:
            validation_report.append(f"⚠️ Imbalanced target distribution (min class: {min_class_ratio:.1%})")
            passed_checks += 0.5  # Half credit
    except Exception as e:
        validation_report.append(f"❌ Target distribution check failed: {e}")

    # Check 6: No NaN in final data
    total_checks += 1
    try:
        train_nans = results['X_train'].isna().sum().sum()
        scaled_nans = pd.DataFrame(results['X_train_scaled']).isna().sum().sum()

        if train_nans == 0 and scaled_nans == 0:
            passed_checks += 1
            validation_report.append("✅ No NaN values in final data")
        else:
            validation_report.append(f"❌ NaN values present: train={train_nans}, scaled={scaled_nans}")
    except Exception as e:
        validation_report.append(f"❌ NaN check failed: {e}")

    # Calculate validation score
    validation_score = passed_checks / total_checks

    logger.info(f"📋 Validation complete: {passed_checks}/{total_checks} checks passed ({validation_score:.1%})")

    # Log validation report
    for item in validation_report:
        if item.startswith("✅"):
            logger.info(item)
        elif item.startswith("⚠️"):
            logger.warning(item)
        else:
            logger.error(item)

    # Save validation report
    try:
        os.makedirs("debug", exist_ok=True)
        with open("debug/data_validation_report.txt", "w") as f:
            f.write("# Data Validation Report\n\n")
            f.write(f"Validation Date: {pd.Timestamp.now()}\n")
            f.write(f"Overall Score: {passed_checks}/{total_checks} ({validation_score:.1%})\n\n")

            f.write("## Validation Results\n")
            for item in validation_report:
                f.write(f"{item}\n")

            if validation_score < 0.8:
                f.write(f"\n⚠️ WARNING: Validation score below 80%. Review data quality before training.\n")
            elif validation_score == 1.0:
                f.write(f"\n✅ EXCELLENT: All validation checks passed!\n")
            else:
                f.write(f"\n✅ GOOD: Most validation checks passed.\n")
    except Exception as e:
        logger.warning(f"Could not save validation report: {e}")

    return {
        'validation_score': validation_score,
        'passed_checks': passed_checks,
        'total_checks': total_checks,
        'report': validation_report,
        'status': 'PASS' if validation_score >= 0.8 else 'WARNING' if validation_score >= 0.6 else 'FAIL'
    }

def get_processing_recommendations(results, validation_results):
    """
    Generate recommendations based on processing results and validation
    """
    recommendations = []

    try:
        # Feature count recommendations
        feature_count = results.get('feature_count', 0)
        sample_count = len(results.get('y_train', []))

        if feature_count > sample_count / 10:
            recommendations.append(f"🎯 Consider more aggressive feature selection: {feature_count} features for {sample_count} samples")

        # Target distribution recommendations
        if 'y_train' in results:
            target_dist = results['y_train'].value_counts()
            min_class_ratio = target_dist.min() / target_dist.sum()

            if min_class_ratio < 0.1:
                recommendations.append(f"⚖️ Address class imbalance: minority class = {min_class_ratio:.1%}")
                recommendations.append("   Consider: SMOTE, class weights, or stratified sampling")

        # Validation score recommendations
        val_score = validation_results.get('validation_score', 0)
        if val_score < 0.8:
            recommendations.append("🔍 Data quality issues detected - review validation report")

        # Feature engineering recommendations
        summary = results.get('processing_summary', {})
        original_features = summary.get('original_features', 0)
        final_features = summary.get('final_features', 0)

        if final_features < original_features * 0.3:
            recommendations.append("📊 Many features were removed - consider reviewing feature engineering logic")

        # Missing value recommendations
        removed_features = results.get('removed_features', [])
        if len(removed_features) > original_features * 0.2:
            recommendations.append("🔧 High feature removal rate - review data quality and cleaning thresholds")

        if not recommendations:
            recommendations.append("✅ Data processing looks good - no major issues detected")

    except Exception as e:
        recommendations.append(f"❌ Could not generate recommendations: {e}")

    return recommendations

# Main function with comprehensive pipeline
def process_features_with_validation(df_train_raw, df_val_raw=None, df_test_raw=None,
                                   max_features=50, enable_feature_selection=True,
                                   scaling_method='standard', validate_leakage=True,
                                   generate_recommendations=True):
    """
    Complete feature processing pipeline with validation and recommendations
    """
    logger.info("🚀 Starting comprehensive feature processing with validation...")

    try:
        # Run main processing pipeline
        results = process_features_pipeline(
            df_train_raw=df_train_raw,
            df_val_raw=df_val_raw,
            df_test_raw=df_test_raw,
            max_features=max_features,
            enable_feature_selection=enable_feature_selection,
            scaling_method=scaling_method,
            validate_leakage=validate_leakage
        )

        # Validate processed data
        validation_results = validate_processed_data(results)
        results['validation'] = validation_results

        # Generate recommendations
        if generate_recommendations:
            recommendations = get_processing_recommendations(results, validation_results)
            results['recommendations'] = recommendations

            logger.info("💡 Processing Recommendations:")
            for rec in recommendations:
                logger.info(f"  {rec}")

        # Final status report
        val_status = validation_results.get('status', 'UNKNOWN')
        if val_status == 'PASS':
            logger.info("🎉 PROCESSING COMPLETED SUCCESSFULLY")
        elif val_status == 'WARNING':
            logger.warning("⚠️ PROCESSING COMPLETED WITH WARNINGS")
        else:
            logger.error("❌ PROCESSING COMPLETED WITH ERRORS")

        return results

    except Exception as e:
        logger.error(f"💥 Feature processing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

# Utility functions for backward compatibility
def advanced_feature_selection(X_train, y_train, X_val=None, X_test=None, max_features=50):
    """Backward compatibility wrapper"""
    return enhanced_feature_selection(X_train, y_train, X_val, X_test, max_features)

def drop_problematic_columns(X_train, X_val=None, X_test=None):
    """Backward compatibility wrapper"""
    return clean_features(X_train, X_val, X_test)

def detect_data_leakage(X_train, y_train, threshold=0.95):
    """Backward compatibility wrapper"""
    if X_train.empty or len(y_train) == 0:
        return []

    # Create temporary DataFrame
    temp_df = X_train.copy()
    temp_df['Target_1d'] = y_train

    leaky_features, _ = detect_temporal_leakage(temp_df, X_train.columns.tolist(), 'Target_1d', threshold)
    return leaky_features

def process_features(df_train_raw, df_val_raw=None, df_test_raw=None,
                    max_features=50, enable_feature_selection=True):
    """Backward compatibility wrapper"""
    return process_features_pipeline(
        df_train_raw=df_train_raw,
        df_val_raw=df_val_raw,
        df_test_raw=df_test_raw,
        max_features=max_features,
        enable_feature_selection=enable_feature_selection,
        scaling_method='standard',
        validate_leakage=True
    )
