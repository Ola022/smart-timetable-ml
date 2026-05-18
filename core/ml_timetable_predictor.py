"""
ML-based Timetable Predictor
Uses pre-trained models (RandomForest, GradientBoosting, LogisticRegression) to predict optimal time slot assignments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import os

logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


class TimetableMLPredictor:
    """
    Machine Learning predictor for timetable scheduling
    Uses pre-trained models for time slot prediction
    """

    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.lr_model = None
        self.label_encoders = {}
        self.feature_columns = [
            "CourseID",
            "LecturerID",
            "LevelText",
            "LevelGroupID",
            "Semester",
            "SlotID",
            "VenueID",
            "LecturerQualified",
            "VenueCapacitySuitable",
            "LecturerAlreadyBooked",
            "VenueAlreadyBooked",
            "LevelAlreadyBooked"
        ]

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data: encode categorical variables, handle missing values"""
        df = df.copy()
        
        # Encode Semester
        if 'Semester' in df.columns:
            if 'Semester' in self.label_encoders:
                df['Semester'] = self.label_encoders['Semester'].transform(df['Semester'])
            else:
                le = LabelEncoder()
                df['Semester'] = le.fit_transform(df['Semester'])
                self.label_encoders['Semester'] = le
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Fill missing values
        df = df.fillna(0)
        
        return df

    def predict_all_models(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict suitability using all available models
        Returns DataFrame with scores from RandomForest, GradientBoosting, and LogisticRegression
        """
        logger.info("Predicting with all available models...")
        
        # Preprocess candidates
        candidates_processed = self.preprocess_data(candidates_df)
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in candidates_processed.columns:
                candidates_processed[col] = 0
        
        X_candidates = candidates_processed[self.feature_columns]
        
        # Add scores to original dataframe
        result_df = candidates_df.copy()
        
        # RandomForest
        if self.rf_model:
            rf_probs = self.rf_model.predict_proba(X_candidates)[:, 1]
            result_df['RF_score'] = rf_probs
            logger.info("Added RandomForest scores")
        else:
            result_df['RF_score'] = 0.0
            logger.warning("RandomForest model not available")
        
        # GradientBoosting
        if self.gb_model:
            gb_probs = self.gb_model.predict_proba(X_candidates)[:, 1]
            result_df['GB_score'] = gb_probs
            logger.info("Added GradientBoosting scores")
        else:
            result_df['GB_score'] = 0.0
            logger.warning("GradientBoosting model not available")
        
        # LogisticRegression
        if self.lr_model:
            lr_probs = self.lr_model.predict_proba(X_candidates)[:, 1]
            result_df['LR_score'] = lr_probs
            logger.info("Added LogisticRegression scores")
        else:
            result_df['LR_score'] = 0.0
            logger.warning("LogisticRegression model not available")
        
        logger.info(f"Generated predictions for {len(result_df)} candidates with all models")
        
        return result_df

    def predict_slot_suitability(
        self,
        candidates_df: pd.DataFrame,
        model_name: str = 'RandomForest',
        threshold: float = 0.75
    ) -> pd.DataFrame:
        """
        Predict suitability of candidate slot assignments
        Returns DataFrame with probability scores and filtered results
        """
        logger.info(f"Predicting with {model_name} model...")
        
        # Preprocess candidates
        candidates_processed = self.preprocess_data(candidates_df)
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in candidates_processed.columns:
                candidates_processed[col] = 0
        
        X_candidates = candidates_processed[self.feature_columns]
        
        # Select model
        if model_name == 'RandomForest' and self.rf_model:
            model = self.rf_model
        elif model_name == 'GradientBoosting' and self.gb_model:
            model = self.gb_model
        elif model_name == 'LogisticRegression' and self.lr_model:
            model = self.lr_model
        else:
            logger.warning(f"Model {model_name} not available, using RandomForest")
            model = self.rf_model

        if model is None:
            raise RuntimeError(f"Requested model '{model_name}' is not loaded. Check ml-models or sklearn compatibility.")
        
        # Get predictions
        probs = model.predict_proba(X_candidates)[:, 1]
        
        # Add scores to original dataframe
        result_df = candidates_df.copy()
        result_df[f'{model_name}_score'] = probs
        result_df['predicted_suitable'] = (probs >= threshold).astype(int)
        
        # Filter high-confidence predictions
        filtered_df = result_df[result_df[f'{model_name}_score'] >= threshold].copy()
        
        logger.info(f"Generated predictions for {len(result_df)} candidates, "
                   f"{len(filtered_df)} passed threshold {threshold}")
        
        return result_df, filtered_df

    def rank_candidates(
        self,
        candidates_df: pd.DataFrame,
        model_name: str = 'RandomForest'
    ) -> pd.DataFrame:
        """
        Rank candidates by model prediction score
        Returns sorted DataFrame
        """
        logger.info(f"=== RANKING CANDIDATES WITH {model_name} MODEL ===")
        logger.info(f"Number of candidates to rank: {len(candidates_df)}")
        
        # Verify model is loaded
        if model_name == 'RandomForest' and not self.rf_model:
            logger.error(f"RandomForest model not loaded!")
        elif model_name == 'GradientBoosting' and not self.gb_model:
            logger.error(f"GradientBoosting model not loaded!")
        elif model_name == 'LogisticRegression' and not self.lr_model:
            logger.error(f"LogisticRegression model not loaded!")
        
        scored_df, _ = self.predict_slot_suitability(candidates_df, model_name, threshold=0.0)
        ranked_df = scored_df.sort_values(f'{model_name}_score', ascending=False)
        
        logger.info(f"Top 3 {model_name} scores: {ranked_df.head(3)[f'{model_name}_score'].tolist()}")
        return ranked_df

    def get_feature_importance(self, model_name: str = 'RandomForest') -> pd.Series:
        """Get feature importance from trained model"""
        if model_name == 'RandomForest' and self.rf_model:
            model = self.rf_model
        elif model_name == 'GradientBoosting' and self.gb_model:
            model = self.gb_model
        elif model_name == 'LogisticRegression' and self.lr_model:
            model = self.lr_model
        else:
            raise ValueError(f"Model {model_name} not available")
        
        importances = pd.Series(model.feature_importances_, index=self.feature_columns)
        return importances.sort_values(ascending=False)

    def load_models(self, directory: str = None) -> None:
        """Load trained models from disk"""
        if directory is None:
            directory = os.path.join(PROJECT_ROOT, "ml-models")
        logger.info(f"Loading models from {directory}")
        
        try:
            self.rf_model = joblib.load(f"{directory}/RandomForest_timetable_model.pkl")
            logger.info("Loaded RandomForest model")
        except FileNotFoundError:
            logger.warning("RandomForest model not found")
            self.rf_model = None
        except Exception as e:
            logger.warning(f"Failed to load RandomForest model: {e}")
            self.rf_model = None
        
        try:
            self.gb_model = joblib.load(f"{directory}/GradientBoosting_timetable_model.pkl")
            logger.info("Loaded GradientBoosting model")
        except FileNotFoundError:
            logger.warning("GradientBoosting model not found")
            self.gb_model = None
        except Exception as e:
            logger.warning(f"Failed to load GradientBoosting model: {e}")
            self.gb_model = None
        
        try:
            self.lr_model = joblib.load(f"{directory}/LogisticRegression_timetable_model.pkl")
            logger.info("Loaded LogisticRegression model")
        except FileNotFoundError:
            logger.warning("LogisticRegression model not found")
            self.lr_model = None
        except Exception as e:
            logger.warning(f"Failed to load LogisticRegression model: {e}")
            self.lr_model = None
        
        try:
            self.label_encoders = joblib.load(f"{directory}/label_encoders.pkl")
            logger.info("Loaded label encoders")
        except FileNotFoundError:
            logger.warning("Label encoders not found")
            self.label_encoders = {}
        except Exception as e:
            logger.warning(f"Failed to load label encoders: {e}")
            self.label_encoders = {}
