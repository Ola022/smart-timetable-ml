"""
FastAPI Backend for Smart Timetable Scheduler UI
Provides endpoints for model selection, scheduling, and timetable management
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import logging
from datetime import datetime

from smart_timetable_scheduler import TimetableScheduler
from ml_timetable_predictor import TimetableMLPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Timetable Scheduler API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ScheduleRequest(BaseModel):
    """Request model for scheduling"""
    model_name: str = "RandomForest"  # RandomForest, GradientBoosting, LogisticRegression, or Basic
    max_attempts: int = 5
    use_ml: bool = True
    use_fallback: bool = True
    threshold: float = 0.75


class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    accuracy: float
    roc_auc: float
    available: bool
    threshold: float = 0.0


class ScheduleResult(BaseModel):
    """Schedule result response"""
    success: bool
    message: str
    total_courses: int
    scheduled_courses: int
    unscheduled_courses: int
    coverage: float
    score: float
    lecturer_load: Dict[str, int]
    venue_utilization: Dict[str, int]
    day_distribution: Dict[str, int]
    conflicts: Dict[str, List[str]]
    total_slots_assigned: int
    timetable: List[Dict]
    unscheduled_list: List[Dict]
    model_used: Optional[str] = None
    execution_time: float


# Global state
predictor = None
model_metrics = {}


@app.on_event("startup")
async def startup_event():
    """Initialize ML predictor on startup"""
    global predictor, model_metrics
    logger.info("Loading ML predictor...")
    try:
        predictor = TimetableMLPredictor()
        predictor.load_models()
        
        # Load training data to get metrics
        try:
            df = predictor.load_training_data("timetable_training_dataset.csv")
            results = predictor.train_models(df, retrain=False)  # Just get metrics, don't retrain
            model_metrics = results
            logger.info(f"Loaded metrics for {len(results)} models")
        except Exception as e:
            logger.warning(f"Could not load model metrics: {e}")
            model_metrics = {
                "RandomForest": {"accuracy": 0.9550, "roc_auc": 0.9392},
                "GradientBoosting": {"accuracy": 0.9425, "roc_auc": 0.9336},
            }
    except Exception as e:
        logger.warning(f"Could not load ML models: {e}")
        predictor = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Timetable Scheduler API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/api/models",
            "schedule": "/api/schedule",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_available": predictor is not None
    }


@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """Get available ML models with their metrics"""
    models = []
    
    # Basic scheduler (no ML)
    models.append(ModelInfo(
        name="Basic",
        accuracy=0.90,
        roc_auc=0.0,
        available=True,
        threshold=0.0
    ))
    
    # ML models
    if predictor and model_metrics:
        for model_name, metrics in model_metrics.items():
            if metrics:
                models.append(ModelInfo(
                    name=model_name,
                    accuracy=metrics.get("accuracy", 0.0),
                    roc_auc=metrics.get("roc_auc", 0.0),
                    available=True,
                    threshold=0.0  # Default threshold
                ))
    
    return models


@app.post("/api/schedule", response_model=ScheduleResult)
async def schedule_timetable(request: ScheduleRequest):
    """Generate timetable using specified model"""
    start_time = datetime.now()
    
    try:
        logger.info(
            f"Received scheduling request: model_name={request.model_name}, use_ml={request.use_ml}, "
            f"max_attempts={request.max_attempts}, use_fallback={request.use_fallback}, threshold={request.threshold}"
        )

        # Initialize scheduler
        scheduler = TimetableScheduler()
        
        # Select ML predictor if requested
        ml_predictor = None
        model_used = "Basic"
        model_name_normalized = request.model_name.strip().lower()
        
        if request.use_ml and predictor:
            if "randomforest" in model_name_normalized:
                ml_predictor = predictor
                model_used = "RandomForest"
            elif "gradientboosting" in model_name_normalized:
                ml_predictor = predictor
                model_used = "GradientBoosting"
            elif "logistic" in model_name_normalized or "logisticregression" in model_name_normalized:
                ml_predictor = predictor
                model_used = "LogisticRegression"
            elif "basic" in model_name_normalized:
                ml_predictor = None
                model_used = "Basic"
            else:
                # Default to RandomForest if specified model not found
                ml_predictor = predictor
                model_used = "RandomForest"
                logger.warning(f"Unknown model_name '{request.model_name}' received, defaulting to RandomForest")
        
        logger.info(f"Selected model_used={model_used}, ml_predictor={'yes' if ml_predictor else 'no'}")

        # Run scheduler
        result, evaluation = scheduler.run(
            max_attempts=request.max_attempts,
            ml_predictor=ml_predictor,
            model_name=model_used,
            use_fallback=request.use_fallback,
            progress_callback=None
        )
        
        # Convert timetable to list of dicts
        timetable_list = result.timetable.to_dict(orient="records")
        
        # Convert unscheduled courses to list of dicts
        unscheduled_list = []
        for failure in result.unscheduled_courses:
            unscheduled_list.append({
                "course_code": failure.course_code,
                "course_id": failure.course_id,
                "reason": failure.reason
            })
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ScheduleResult(
            success=True,
            message="Timetable generated successfully",
            total_courses=evaluation["total_courses"],
            scheduled_courses=evaluation["scheduled_courses"],
            unscheduled_courses=evaluation["unscheduled_courses"],
            coverage=evaluation["coverage"],
            score=evaluation["score"],
            lecturer_load=evaluation["lecturer_load"],
            venue_utilization=evaluation["venue_utilization"],
            day_distribution=evaluation["day_distribution"],
            conflicts=evaluation["conflicts"],
            total_slots_assigned=evaluation["total_slots_assigned"],
            timetable=timetable_list,
            unscheduled_list=unscheduled_list,
            model_used=model_used,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timetable/latest")
async def get_latest_timetable():
    """Get the latest generated timetable"""
    try:
        # Try to load the ML-guided timetable
        try:
            df = pd.read_csv("ml_guided_timetable.csv")
            return {
                "success": True,
                "timetable": df.to_dict(orient="records"),
                "rows": len(df),
                "source": "ml_guided_timetable.csv"
            }
        except FileNotFoundError:
            # Try basic timetable
            try:
                df = pd.read_csv("test_basic_timetable.csv")
                return {
                    "success": True,
                    "timetable": df.to_dict(orient="records"),
                    "rows": len(df),
                    "source": "test_basic_timetable.csv"
                }
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="No timetable found")
    except Exception as e:
        logger.error(f"Failed to load timetable: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/courses")
async def get_courses():
    """Get list of courses"""
    try:
        df = pd.read_csv("courses.csv")
        return {
            "success": True,
            "courses": df.to_dict(orient="records"),
            "total": len(df)
        }
    except Exception as e:
        logger.error(f"Failed to load courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/lecturers")
async def get_lecturers():
    """Get list of lecturers"""
    try:
        df = pd.read_csv("lecturers.csv")
        return {
            "success": True,
            "lecturers": df.to_dict(orient="records"),
            "total": len(df)
        }
    except Exception as e:
        logger.error(f"Failed to load lecturers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/venues")
async def get_venues():
    """Get list of venues"""
    try:
        df = pd.read_csv("venues.csv")
        return {
            "success": True,
            "venues": df.to_dict(orient="records"),
            "total": len(df)
        }
    except Exception as e:
        logger.error(f"Failed to load venues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/timeslots")
async def get_timeslots():
    """Get list of time slots"""
    try:
        df = pd.read_csv("timeslots.csv")
        return {
            "success": True,
            "timeslots": df.to_dict(orient="records"),
            "total": len(df)
        }
    except Exception as e:
        logger.error(f"Failed to load timeslots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
