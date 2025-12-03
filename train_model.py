"""
F1 Race Predictor - Model Training Script
Trains ensemble models and saves artifacts for deployment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime
from typing import Dict, Tuple

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

from config import FEATURE_COLUMNS, MODEL_CONFIG, ENSEMBLE_WEIGHTS, TRAINING_SEASONS
from data_loader import F1DataLoader, save_raw_data
from feature_builder import FeatureBuilder
from data_tracker import validate_data_completeness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1ModelTrainer:
    """
    Handles model training, evaluation, and artifact saving.
    """

    def __init__(self):
        self.models = {}
        self.imputer = None
        self.feature_builder = None
        self.training_info = {}

    def prepare_training_data(self, seasons: list = TRAINING_SEASONS) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data with complete tracking.

        Returns:
            (X, y): Features and target
        """
        logger.info("="*60)
        logger.info("STEP 1: Loading Race Data (with tracking)")
        logger.info("="*60)

        # Load raw data (this will track all attempts and validate completeness)
        loader = F1DataLoader()
        raw_data = loader.load_all_race_data(seasons, retry_failed=True)

        if raw_data.empty:
            logger.error("\n❌ NO DATA LOADED!")
            logger.error("Check the race loading report for details:")
            logger.error("  cat race_loading_report.csv")
            logger.error("\nTo retry failed races:")
            logger.error("  python validate_and_retry.py retry")
            raise ValueError("No data loaded! Check FastF1 connectivity and race_loading_report.csv")

        # Check if data is complete
        from data_tracker import validate_data_completeness
        is_complete, issues = validate_data_completeness(loader.tracker, seasons)

        if not is_complete:
            logger.warning(f"\n⚠ WARNING: Training with incomplete data ({len(issues)} races missing)")
            logger.warning("Model may have reduced accuracy. To retry missing races:")
            logger.warning("  python validate_and_retry.py retry")

            # Ask user if they want to proceed
            response = input("\nProceed with training anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                logger.info("Training cancelled. Fix data issues first.")
                raise ValueError("Incomplete training data - user cancelled")

        # Save backup
        save_raw_data(raw_data, 'training_data_raw.csv')

        logger.info("\n" + "="*60)
        logger.info("STEP 2: Building Features")
        logger.info("="*60)

        # Build features
        self.feature_builder = FeatureBuilder(raw_data)
        features_df = self.feature_builder.build_features()

        # Filter out rows without target (shouldn't happen, but safety check)
        features_df = features_df.dropna(subset=['ActualPosition'])

        # Split features and target
        X = features_df[FEATURE_COLUMNS]
        y = features_df['ActualPosition']

        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target range: {y.min():.1f} to {y.max():.1f}")

        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Train all models: RandomForest, XGBoost (if available), LinearRegression.
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Training Models")
        logger.info("="*60)

        # Impute missing values (should be minimal, but handle it)
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)

        logger.info(f"Imputed {np.isnan(X.values).sum()} missing values")

        # Train RandomForest
        logger.info("\nTraining Random Forest...")
        rf = RandomForestRegressor(**MODEL_CONFIG['random_forest'])
        rf.fit(X_imputed, y)
        self.models['rf'] = rf
        logger.info("✓ Random Forest trained")

        # Train XGBoost (if available)
        if XGBOOST_AVAILABLE:
            logger.info("\nTraining XGBoost...")
            xgb = XGBRegressor(**MODEL_CONFIG['xgboost'])
            xgb.fit(X_imputed, y)
            self.models['xgb'] = xgb
            logger.info("✓ XGBoost trained")
        else:
            logger.warning("⚠ XGBoost not available, skipping")

        # Train Linear Regression
        logger.info("\nTraining Linear Regression...")
        lr = LinearRegression(**MODEL_CONFIG['linear'])
        lr.fit(X_imputed, y)
        self.models['lr'] = lr
        logger.info("✓ Linear Regression trained")

    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate model performance on training data.

        Note: This is IN-SAMPLE evaluation. For proper validation,
        use hold-out set or cross-validation.
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Evaluating Models")
        logger.info("="*60)

        X_imputed = self.imputer.transform(X)
        results = {}

        for name, model in self.models.items():
            preds = model.predict(X_imputed)
            preds = np.clip(preds, 1, 20)  # Clip to valid position range

            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)

            results[name] = {
                'mae': mae,
                'r2': r2,
                'predictions': preds
            }

            logger.info(f"{name.upper()}: MAE={mae:.2f}, R²={r2:.3f}")

        # Ensemble prediction
        if 'xgb' in results:
            ensemble = (
                ENSEMBLE_WEIGHTS['rf'] * results['rf']['predictions'] +
                ENSEMBLE_WEIGHTS['xgb'] * results['xgb']['predictions'] +
                ENSEMBLE_WEIGHTS['lr'] * results['lr']['predictions']
            )
        else:
            # Fallback if no XGBoost: 50% RF, 50% LR
            ensemble = (
                0.5 * results['rf']['predictions'] +
                0.5 * results['lr']['predictions']
            )

        ensemble = np.clip(ensemble, 1, 20)
        ensemble_mae = mean_absolute_error(y, ensemble)
        ensemble_r2 = r2_score(y, ensemble)

        results['ensemble'] = {
            'mae': ensemble_mae,
            'r2': ensemble_r2,
            'predictions': ensemble
        }

        logger.info(f"ENSEMBLE: MAE={ensemble_mae:.2f}, R²={ensemble_r2:.3f}")

        return results

    def save_artifacts(self, filepath: str = 'f1_model_artifacts.pkl'):
        """
        Save all trained models and metadata to disk.

        Artifacts include:
        - Trained models (rf, xgb, lr)
        - Imputer
        - Feature builder (with stats and encoding maps)
        - Training metadata
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Saving Artifacts")
        logger.info("="*60)

        artifacts = {
            'models': self.models,
            'imputer': self.imputer,
            'feature_columns': FEATURE_COLUMNS,
            'feature_builder': {
                'driver_stats': self.feature_builder.driver_stats,
                'team_stats': self.feature_builder.team_stats,
                'circuit_stats': self.feature_builder.circuit_stats,
                'driver_map': self.feature_builder.driver_map,
                'team_map': self.feature_builder.team_map,
                'circuit_map': self.feature_builder.circuit_map,
            },
            'training_info': {
                'training_date': datetime.now().isoformat(),
                'training_seasons': TRAINING_SEASONS,
                'xgboost_available': XGBOOST_AVAILABLE,
            }
        }

        joblib.dump(artifacts, filepath)
        logger.info(f"✓ Artifacts saved to {filepath}")
        logger.info(f"  File size: {joblib.os.path.getsize(filepath) / 1024 / 1024:.1f} MB")

    def full_training_pipeline(self):
        """
        Execute complete training pipeline.
        """
        logger.info("\n" + "="*60)
        logger.info("F1 RACE OUTCOME PREDICTOR - TRAINING PIPELINE")
        logger.info("="*60 + "\n")

        try:
            # Prepare data
            X, y = self.prepare_training_data()

            # Train models
            self.train_models(X, y)

            # Evaluate
            results = self.evaluate_models(X, y)

            # Save
            self.save_artifacts()

            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETE!")
            logger.info("="*60)
            logger.info("\nNext steps:")
            logger.info("1. Run the Streamlit app: streamlit run app.py")
            logger.info("2. Select 'Past Race Analysis' to validate predictions")
            logger.info("3. Select 'Future Race Prediction' to predict upcoming races")
            logger.info("\n" + "="*60 + "\n")

            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def load_trained_model(filepath: str = 'f1_model_artifacts.pkl') -> Dict:
    """
    Load previously trained model artifacts.

    Returns:
        Dict containing models, imputer, feature_builder, etc.
    """
    artifacts = joblib.load(filepath)
    logger.info(f"Loaded model artifacts from {filepath}")
    logger.info(f"  Training date: {artifacts['training_info']['training_date']}")
    logger.info(f"  Training seasons: {artifacts['training_info']['training_seasons']}")
    return artifacts


# ==========================
# MAIN EXECUTION
# ==========================

if __name__ == '__main__':
    """
    Run this script to train models and save artifacts.

    Usage:
        python train_model.py

    This will:
    1. Fetch data from FastF1 for 2022-2024 seasons
    2. Build features with edge case handling
    3. Train RandomForest, XGBoost, and LinearRegression
    4. Evaluate ensemble performance
    5. Save artifacts to f1_model_artifacts.pkl
    """

    trainer = F1ModelTrainer()
    results = trainer.full_training_pipeline()

    # Print feature importance (if desired)
    if 'rf' in trainer.models:
        logger.info("\n" + "="*60)
        logger.info("TOP 10 MOST IMPORTANT FEATURES")
        logger.info("="*60)

        rf_model = trainer.models['rf']
        feature_importance = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(feature_importance.head(10).to_string(index=False))