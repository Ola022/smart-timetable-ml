# Smart Timetable Scheduler

A production-grade constraint satisfaction system for academic timetable scheduling with ML-guided optimization.

## Overview

This system solves the university timetable scheduling problem using:
- **Constraint Satisfaction Problem (CSP)** approach with systematic search
- **Machine Learning** integration (RandomForest, GradientBoosting) for intelligent slot assignment
- **Comprehensive validation** ensuring all courses are scheduled
- **Quality scoring** for timetable evaluation
- **Two modes**: Standalone script OR Web UI

## Problem Solved

The original system had critical issues:
- **Unstable scheduling**: 16-17 out of 106 courses missed each run
- **Random allocation**: No systematic search, leading to inconsistent results
- **Silent failures**: Courses skipped without logging reasons
- **No coverage validation**: Missing assertion that all courses must be scheduled
- **Weak constraint handling**: No backtracking or fallback strategies

## Solution Features

### 1. Constraint Satisfaction Algorithm
- Systematic search through slot/venue/lecturer combinations
- Multiple fallback strategies when constraints are tight
- Level conflict relaxation as last resort
- Guaranteed course scheduling through retry mechanisms

### 2. Hard Constraints Enforced
- **No lecturer double-booking**: Each lecturer teaches max one class per slot
- **No venue double-booking**: Each venue hosts max one class per slot
- **Capacity requirements**: Venue capacity >= expected students
- **Lecturer qualification**: Lecturers only teach levels they're qualified for
- **Weekly limits**: Each course scheduled for exactly 2 sessions

### 3. Soft Constraints
- **Level conflicts**: Minimize same-level courses at same time
- **Lecturer load balance**: Distribute teaching load evenly
- **Venue utilization**: Efficient use of available spaces

## File Descriptions

### Core Python Files (core/)

**smart_timetable_scheduler.py**
- Main scheduler implementation using constraint satisfaction with backtracking
- Handles lecturer qualification, venue capacity, and conflict checking
- Supports both basic constraint-based and ML-guided scheduling
- Generates ML datasets for training

**ml_timetable_predictor.py**
- ML model training (RandomForest, GradientBoosting)
- Model loading and prediction
- Candidate ranking for ML-guided scheduling
- Feature importance analysis

**run_scheduler.py**
- Standalone CLI script - no server needed
- Interactive model selection
- Generates timetable CSV and HTML report
- Perfect for quick use without web UI

**app.py**
- FastAPI backend for web UI
- REST API endpoints for model selection and scheduling
- Serves timetable data and metrics

### Data Files (data_files/)

**courses.csv** - Course information (CourseID, CourseCode, LecturerID, Level, Semester, Department)
**lecturers.csv** - Lecturer information (LecturerID, Name, Department, Rank)
**venues.csv** - Venue information (VenueID, Name, Capacity)
**timeslots.csv** - Time slot information (SlotID, Day, TimeRange)

### ML Models (ml-models/)

**RandomForest_timetable_model.pkl** - Trained RandomForest model (95.5% accuracy)
**GradientBoosting_timetable_model.pkl** - Trained GradientBoosting model (94.25% accuracy)
**label_encoders.pkl** - Encoders for categorical features

### Notebook (notebook/)

**Untitled1.ipynb** - Working notebook for training ML models and testing the scheduler

### Config Files (config/)

**requirements.txt** - Python dependencies
**README.md** - This documentation
**index.html** - Web UI interface

## Installation

```bash
pip install -r config/requirements.txt
```

## Quick Start

### Option 1: Standalone Script (No Server)

Run the standalone script for quick scheduling:

```bash
python core/run_scheduler.py
```

Select scheduling method:
1. Basic (Constraint-based only) - Fast, no ML
2. RandomForest (ML-guided) - Best accuracy (95.5%)
3. GradientBoosting (ML-guided) - Good accuracy (94.25%)

The script will:
- Generate timetable
- Display results in terminal
- Save CSV files
- Generate and open HTML report

### Option 2: Web UI (Interactive)

Start the FastAPI server:

```bash
python core/app.py
```

Then open `config/index.html` in your browser.

## Data Files Required

- `courses.csv` - Course information (ID, code, lecturer, level, department, semester)
- `lecturers.csv` - Lecturer information (ID, name, department, rank)
- `venues.csv` - Venue information (ID, name, capacity)
- `timeslots.csv` - Time slot information (ID, day, time range)

## Web UI API Documentation

### Start API Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Endpoints

#### GET `/api/models`
Get available ML models with their metrics (accuracy, ROC AUC)

**Response:**
```json
[
  {
    "name": "RandomForest",
    "accuracy": 0.9550,
    "roc_auc": 0.9392,
    "available": true,
    "threshold": 0.0
  },
  {
    "name": "GradientBoosting",
    "accuracy": 0.9425,
    "roc_auc": 0.9336,
    "available": true,
    "threshold": 0.0
  },
  {
    "name": "Basic",
    "accuracy": 0.90,
    "roc_auc": 0.0,
    "available": true,
    "threshold": 0.0
  }
]
```

#### POST `/api/schedule`
Generate timetable using specified model

**Request:**
```json
{
  "model_name": "RandomForest",
  "max_attempts": 5,
  "use_ml": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Timetable generated successfully",
  "total_courses": 106,
  "scheduled_courses": 106,
  "unscheduled_courses": 0,
  "coverage": 1.0,
  "score": 0.9460,
  "lecturer_load": {...},
  "venue_utilization": {...},
  "day_distribution": {...},
  "conflicts": {},
  "total_slots_assigned": 212,
  "timetable": [...],
  "unscheduled_list": [],
  "model_used": "RandomForest",
  "execution_time": 45.2
}
```

#### GET `/api/timetable/latest`
Get the latest generated timetable

#### GET `/api/data/courses`
Get list of courses

#### GET `/api/data/lecturers`
Get list of lecturers

#### GET `/api/data/venues`
Get list of venues

#### GET `/api/data/timeslots`
Get list of time slots

## Module Structure

### `smart_timetable_scheduler.py`

Main scheduler module implementing constraint satisfaction:

- **TimetableScheduler**: Core scheduling class
  - `load_data()`: Load CSV files
  - `validate_inputs()`: Validate data integrity
  - `build_lookups()`: Build efficient lookup dictionaries
  - `find_valid_assignment()`: Systematic search for valid assignments
  - `schedule_courses()`: Main scheduling algorithm
  - `resolve_conflicts()`: Conflict resolution with backtracking
  - `evaluate_schedule()`: Quality scoring and metrics
  - `generate_ml_dataset()`: Generate ML training data
  - `run()`: Main execution method

### `ml_timetable_predictor.py`

ML module for time slot prediction:

- **TimetableMLPredictor**: ML predictor class
  - `load_training_data()`: Load training dataset
  - `preprocess_data()`: Encode categorical variables
  - `train_models()`: Train RandomForest, GradientBoosting, XGBoost
  - `predict_slot_suitability()`: Score candidate assignments
  - `rank_candidates()`: Rank by ML score
  - `get_feature_importance()`: Feature importance analysis
  - `save_models()` / `load_models()`: Model persistence

### `test_scheduler.py`

Comprehensive test suite:

- `test_basic_scheduler()`: Test constraint-based scheduling
- `test_ml_guided_scheduler()`: Test ML-guided scheduling
- `compare_schedulers()`: Compare both approaches
- `validate_constraints()`: Validate hard constraints

## Output Files

- `final_timetable.csv` - Final scheduled timetable
- `ml_training_dataset.csv` - ML-ready training data
- `test_basic_timetable.csv` - Test output from basic scheduler
- `test_ml_guided_timetable.csv` - Test output from ML scheduler
- `RandomForest_timetable_model.pkl` - Trained RandomForest model
- `GradientBoosting_timetable_model.pkl` - Trained GradientBoosting model
- `XGBoost_timetable_model.pkl` - Trained XGBoost model (optional)

## Scoring System

Timetables are scored based on:
- **Coverage**: Percentage of courses scheduled (primary)
- **Conflicts**: Penalty for level conflicts
- **Load balance**: Variance in lecturer teaching load
- **Utilization**: Venue usage efficiency

Score range: 0.0 to 1.0 (higher is better)

## Algorithm Details

### Constraint Satisfaction Approach

1. **Course ordering**: Higher levels first (fewer students, more flexibility)
2. **Candidate generation**: All valid slot/venue/lecturer combinations
3. **ML ranking**: If ML available, rank candidates by predicted suitability
4. **Systematic selection**: Try highest-ranked conflict-free candidates first
5. **Fallback strategies**: Relax level conflict if no other options
6. **Conflict resolution**: Retry unscheduled courses with extended attempts

### Hard Constraints (Never Violated)

- Lecturer teaches max one class per slot
- Venue hosts max one class per slot
- Venue capacity >= course student count
- Lecturer qualified for course level
- Each course gets exactly 2 sessions

### Soft Constraints (Minimized)

- Same-level courses at same time
- Uneven lecturer load distribution
- Poor venue utilization
- Clustering on same days

## Performance

- **106 courses** scheduled in **~2-5 seconds**
- **100% coverage** guaranteed through assertion
- **Deterministic** results with same seed
- **Scalable** to larger datasets

## Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

scheduler = TimetableScheduler(data_dir=".")
result, evaluation = scheduler.run(max_attempts=5)

# View debug logs
for log in result.debug_logs:
    print(log)
```

## Troubleshooting

### Courses not scheduling

1. Check debug logs for specific failure reasons
2. Verify lecturer qualifications in `lecturers.csv`
3. Check venue capacities in `venues.csv`
4. Ensure sufficient time slots in `timeslots.csv`
5. Review constraint violations in evaluation

### ML models not improving results

1. Ensure training data is representative
2. Check feature importance for insights
3. Try different model hyperparameters
4. Generate more training data from successful schedules

### Coverage validation fails

1. Review unscheduled courses in result
2. Check if constraints are too strict
3. Verify data integrity (no missing lecturers/venues)
4. Increase `max_attempts` parameter

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost (optional)

## License

This is a production-grade scheduling system for academic institutions.

## Author

Smart Timetable Scheduler - Constraint Satisfaction with ML Integration
