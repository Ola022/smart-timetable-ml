"""
Smart Timetable Scheduler
Constraint Satisfaction Problem solver with ML-guided optimization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Set, Optional, Tuple, NamedTuple
from collections import defaultdict, Counter
from enum import Enum
from dataclasses import dataclass, field
import random
import os

logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints for scheduling"""
    LECTURER_QUALIFIED = "lecturer_qualified"
    VENUE_CAPACITY = "venue_capacity"
    LECTURER_CONFLICT = "lecturer_conflict"
    VENUE_CONFLICT = "venue_conflict"
    LEVEL_CONFLICT = "level_conflict"
    WEEKLY_LIMIT = "weekly_limit"


@dataclass
class SchedulingFailure:
    """Record why a course failed to schedule"""
    course_id: int
    course_code: str
    reason: str
    constraint_violated: ConstraintType
    attempted_slots: List[int] = field(default_factory=list)
    attempted_venues: List[int] = field(default_factory=list)
    attempted_lecturers: List[int] = field(default_factory=list)


@dataclass
class ScheduleResult:
    """Result of scheduling operation"""
    timetable: pd.DataFrame
    scheduled_courses: Set[int]
    unscheduled_courses: List[SchedulingFailure]
    conflicts_encountered: Dict[str, int]
    score: float
    debug_logs: List[str]


class TimetableScheduler:
    """
    Constraint Satisfaction Problem solver for timetable scheduling
    Uses systematic search with backtracking and fallback strategies
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir if data_dir else os.path.join(PROJECT_ROOT, "data_files")
        
        # Data containers
        self.courses_df: pd.DataFrame = None
        self.lecturers_df: pd.DataFrame = None
        self.venues_df: pd.DataFrame = None
        self.slots_df: pd.DataFrame = None
        
        # Lookup dictionaries
        self.lecturer_rank: Dict[int, str] = {}
        self.venue_capacity: Dict[int, int] = {}
        self.course_level: Dict[int, int] = {}
        self.course_semester: Dict[int, str] = {}
        self.course_lecturer: Dict[int, int] = {}
        self.course_levelgroup: Dict[int, int] = {}
        self.slot_day: Dict[int, str] = {}
        
        # Configuration
        self.qualification_rules = {
            "Junior": [100, 200],
            "Mid": [200, 300, 400, 500],
            "Senior": [200, 300, 400, 500]
        }
        
        self.level_sizes = {
            100: 250,
            200: 200,
            300: 150,
            400: 120,
            500: 80
        }
        
        self.max_sessions_per_course = 2
        
        # Schedule state
        self.lecturer_schedule: Set[Tuple[int, str, int]] = set()
        self.venue_schedule: Set[Tuple[int, str, int]] = set()
        self.level_schedule: Set[Tuple[int, str, int]] = set()
        self.course_slots: Dict[int, Set[Tuple[str, int]]] = defaultdict(set)
        self.course_session_count: Dict[int, int] = defaultdict(int)

    def load_data(self) -> None:
        """Load all required data from CSV files"""
        logger.info("Loading data files...")
        
        try:
            self.courses_df = pd.read_csv(f"{self.data_dir}/courses.csv")
            self.lecturers_df = pd.read_csv(f"{self.data_dir}/lecturers.csv")
            self.venues_df = pd.read_csv(f"{self.data_dir}/venues.csv")
            self.slots_df = pd.read_csv(f"{self.data_dir}/timeslots.csv")
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV files: {e}")
            raise
        
        logger.info(f"Courses columns: {list(self.courses_df.columns)}")
        logger.info(f"Lecturers columns: {list(self.lecturers_df.columns)}")
        logger.info(f"Venues columns: {list(self.venues_df.columns)}")
        logger.info(f"Slots columns: {list(self.slots_df.columns)}")
        
        # Remove empty rows if any
        self.courses_df = self.courses_df.dropna(subset=['CourseID'])
        self.lecturers_df = self.lecturers_df.dropna(subset=['LecturerID'])
        self.venues_df = self.venues_df.dropna(subset=['VenueID'])
        self.slots_df = self.slots_df.dropna(subset=['SlotID'])
        
        # Convert to appropriate types with error handling
        try:
            self.courses_df['CourseID'] = self.courses_df['CourseID'].astype(int)
            self.courses_df['LecturerID'] = self.courses_df['LecturerID'].astype(int)
            # Handle LevelText - could be already numeric or text
            if self.courses_df['LevelText'].dtype == 'object':
                try:
                    self.courses_df['LevelText'] = self.courses_df['LevelText'].astype(int)
                except ValueError:
                    logger.warning(f"LevelText contains non-numeric values: {self.courses_df['LevelText'].unique()[:10]}")
            self.courses_df['LevelGroupID'] = self.courses_df['LevelGroupID'].astype(int)
            self.lecturers_df['LecturerID'] = self.lecturers_df['LecturerID'].astype(int)
            self.venues_df['VenueID'] = self.venues_df['VenueID'].astype(int)
            self.venues_df['Capacity'] = self.venues_df['Capacity'].astype(int)
            self.slots_df['SlotID'] = self.slots_df['SlotID'].astype(int)
        except Exception as e:
            logger.error(f"Error converting data types: {e}")
            raise
        
        logger.info(f"Loaded {len(self.courses_df)} courses, {len(self.lecturers_df)} lecturers, "
                   f"{len(self.venues_df)} venues, {len(self.slots_df)} slots")

    def validate_inputs(self) -> bool:
        """Validate input data integrity"""
        logger.info("Validating input data...")
        
        errors = []
        
        # Check for missing required columns
        required_course_cols = ['CourseID', 'CourseCode', 'LecturerID', 'LevelText', 'LevelGroupID', 'Department', 'Semester']
        missing_cols = [col for col in required_course_cols if col not in self.courses_df.columns]
        if missing_cols:
            errors.append(f"Missing columns in courses: {missing_cols}. Available: {list(self.courses_df.columns)}")
        
        required_lecturer_cols = ['LecturerID', 'LecturerName', 'Department', 'Rank']
        missing_cols = [col for col in required_lecturer_cols if col not in self.lecturers_df.columns]
        if missing_cols:
            errors.append(f"Missing columns in lecturers: {missing_cols}. Available: {list(self.lecturers_df.columns)}")
        
        required_venue_cols = ['VenueID', 'VenueName', 'Capacity']
        missing_cols = [col for col in required_venue_cols if col not in self.venues_df.columns]
        if missing_cols:
            errors.append(f"Missing columns in venues: {missing_cols}. Available: {list(self.venues_df.columns)}")
        
        required_slot_cols = ['SlotID', 'Day', 'TimeRange']
        missing_cols = [col for col in required_slot_cols if col not in self.slots_df.columns]
        if missing_cols:
            errors.append(f"Missing columns in slots: {missing_cols}. Available: {list(self.slots_df.columns)}")
        
        # Only proceed with other checks if columns exist
        if not errors:
            # Check for empty datasets
            if len(self.courses_df) == 0:
                errors.append("Courses dataset is empty")
            if len(self.lecturers_df) == 0:
                errors.append("Lecturers dataset is empty")
            if len(self.venues_df) == 0:
                errors.append("Venues dataset is empty")
            if len(self.slots_df) == 0:
                errors.append("Slots dataset is empty")
            
            # Check for duplicate course IDs
            if self.courses_df['CourseID'].duplicated().any():
                dup_ids = self.courses_df[self.courses_df['CourseID'].duplicated()]['CourseID'].unique()
                errors.append(f"Duplicate CourseIDs found: {list(dup_ids)}")
            
            # Check lecturer references
            invalid_lecturers = set(self.courses_df['LecturerID']) - set(self.lecturers_df['LecturerID'])
            if invalid_lecturers:
                logger.warning(f"Courses reference invalid lecturers: {invalid_lecturers}")
                errors.append(f"Invalid lecturer references: {invalid_lecturers}. Valid lecturer IDs: {sorted(set(self.lecturers_df['LecturerID']))[:10]}...")
            
            # Check level references
            invalid_levels = set(self.courses_df['LevelText']) - set(self.level_sizes.keys())
            if invalid_levels:
                logger.warning(f"Invalid levels found: {invalid_levels}. Expected levels: {list(self.level_sizes.keys())}")
                errors.append(f"Invalid levels found: {invalid_levels}. Expected: {list(self.level_sizes.keys())}")
        
        if errors:
            for error in errors:
                logger.error(error)
            return False
        
        logger.info("Input validation passed")
        return True

    def build_lookups(self) -> None:
        """Build lookup dictionaries for efficient access"""
        logger.info("Building lookup dictionaries...")
        
        self.lecturer_rank = dict(zip(self.lecturers_df.LecturerID, self.lecturers_df.Rank))
        self.venue_capacity = dict(zip(self.venues_df.VenueID, self.venues_df.Capacity))
        self.course_level = dict(zip(self.courses_df.CourseID, self.courses_df.LevelText))
        self.course_semester = dict(zip(self.courses_df.CourseID, self.courses_df.Semester))
        self.course_lecturer = dict(zip(self.courses_df.CourseID, self.courses_df.LecturerID))
        self.course_levelgroup = dict(zip(self.courses_df.CourseID, self.courses_df.LevelGroupID))
        self.slot_day = dict(zip(self.slots_df.SlotID, self.slots_df.Day))

    def reset_schedule_state(self) -> None:
        """Reset schedule state for new scheduling attempt"""
        self.lecturer_schedule.clear()
        self.venue_schedule.clear()
        self.level_schedule.clear()
        self.course_slots.clear()
        self.course_session_count.clear()

    def get_slot_utilization(self, semester: str) -> Dict[int, int]:
        """Get count of how many times each slot is used"""
        utilization = Counter()
        for (lec_id, sem, slot_id) in self.lecturer_schedule:
            if sem == semester:
                utilization[slot_id] += 1
        return dict(utilization)

    def get_qualified_lecturers(self, level: int) -> List[int]:
        """Get list of lecturers qualified to teach a level"""
        qualified = []
        for lecturer_id, rank in self.lecturer_rank.items():
            if level in self.qualification_rules.get(rank, []):
                qualified.append(lecturer_id)
        return qualified

    def check_lecturer_conflict(self, lecturer_id: int, semester: str, slot_id: int) -> bool:
        """Check if lecturer is already booked at this slot"""
        return (lecturer_id, semester, slot_id) in self.lecturer_schedule

    def check_venue_conflict(self, venue_id: int, semester: str, slot_id: int) -> bool:
        """Check if venue is already booked at this slot"""
        return (venue_id, semester, slot_id) in self.venue_schedule

    def check_level_conflict(self, level: int, semester: str, slot_id: int) -> bool:
        """Check if level already has class at this slot"""
        return (level, semester, slot_id) in self.level_schedule

    def check_venue_capacity(self, venue_id: int, level: int) -> bool:
        """Check if venue has sufficient capacity for level"""
        return self.venue_capacity[venue_id] >= self.level_sizes[level]

    def generate_all_candidates(
        self,
        courses_df: pd.DataFrame,
        slots_df: pd.DataFrame,
        venues_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate all possible candidate allocations following notebook procedure
        Constraint features are set BEFORE ML scoring:
        - LecturerQualified: 1 (lecturer assigned to course)
        - VenueCapacitySuitable: based on actual capacity check
        - LecturerAlreadyBooked: 0 (no bookings yet)
        - VenueAlreadyBooked: 0 (no bookings yet)
        - LevelAlreadyBooked: 0 (no bookings yet)
        """
        candidate_rows = []
        
        for _, course in courses_df.iterrows():
            level = course["LevelText"]
            expected_students = self.level_sizes.get(level, 100)
            
            for _, slot in slots_df.iterrows():
                for _, venue in venues_df.iterrows():
                    venue_capacity = self.venue_capacity.get(venue["VenueID"], 0)
                    capacity_suitable = 1 if venue_capacity >= expected_students else 0
                    
                    row = {
                        "CourseID": course["CourseID"],
                        "LecturerID": course["LecturerID"],
                        "LevelText": course["LevelText"],
                        "LevelGroupID": course["LevelGroupID"],
                        "Semester": course["Semester"],
                        "SlotID": slot["SlotID"],
                        "VenueID": venue["VenueID"],
                        "LecturerQualified": 1,  # Lecturer is assigned to this course
                        "VenueCapacitySuitable": capacity_suitable,  # Based on actual capacity
                        "LecturerAlreadyBooked": 0,  # No bookings yet
                        "VenueAlreadyBooked": 0,  # No bookings yet
                        "LevelAlreadyBooked": 0  # No bookings yet
                    }
                    candidate_rows.append(row)
        
        candidate_df = pd.DataFrame(candidate_rows)
        logger.info(f"Generated {len(candidate_df)} candidate allocations")
        return candidate_df

    def schedule_with_ml_pipeline(
        self,
        ml_predictor,
        model_name: str = 'RandomForest',
        threshold: float = 0.75,
        shuffle: bool = True,
        use_fallback: bool = True,
        progress_callback=None
    ) -> ScheduleResult:
        """
        Schedule using ML pipeline following notebook procedure:
        1. Iterate through courses in random order
        2. For each course, generate candidates with current schedule state
        3. Score candidates with ML model
        4. Pick best candidate that passes constraints
        5. Update schedule state and continue
        """
        logger.info(f"=== SCHEDULING WITH {model_name} MODEL (THRESHOLD: {threshold}) ===")
        
        if progress_callback:
            progress_callback("Resetting schedule state...")
        
        # Reset schedule state
        self.reset_schedule_state()
        
        if progress_callback:
            progress_callback("Shuffling course order...")
        
        # Shuffle course order for variety
        courses_to_schedule = self.courses_df.sample(frac=1).reset_index(drop=True)
        
        timetable = []
        scheduled_course_ids = set()
        all_candidates = []
        
        total_courses = len(courses_to_schedule)
        
        for idx, course in courses_to_schedule.iterrows():
            course_id = course['CourseID']
            level = course['LevelText']
            semester = course['Semester']
            level_group = course['LevelGroupID']
            
            # Update progress
            if progress_callback and idx % 10 == 0:
                progress_callback(f"Scheduling course {idx + 1}/{total_courses} ({len(scheduled_course_ids)} scheduled so far)...")
            
            # Skip if course already scheduled
            if course_id in scheduled_course_ids:
                continue
            
            # Generate candidates for this course with current schedule state
            course_candidates = self._generate_candidates_for_course_incremental(
                course_id, level, semester, level_group
            )
            
            if len(course_candidates) == 0:
                continue
            
            # Add to all candidates for saving
            all_candidates.extend(course_candidates)
            
            # Score candidates with selected ML model only
            candidates_df = pd.DataFrame(course_candidates)
            scored_df, _ = ml_predictor.predict_slot_suitability(candidates_df, model_name, threshold=0.0)
            
            # Use score column based on model
            score_column = f'{model_name}_score'
            
            # Sort by score descending and pick best
            scored_df = scored_df.sort_values(score_column, ascending=False)
            best_candidate = scored_df.iloc[0]
            
            # Check hard constraints
            if self._check_constraints_for_row(best_candidate, timetable):
                timetable.append(best_candidate)
                scheduled_course_ids.add(course_id)
                
                # Update schedule state
                self.lecturer_schedule.add((best_candidate['LecturerID'], semester, best_candidate['SlotID']))
                self.venue_schedule.add((best_candidate['VenueID'], semester, best_candidate['SlotID']))
                if best_candidate['LevelAlreadyBooked'] == 0:
                    self.level_schedule.add((level, semester, best_candidate['SlotID']))
                self.course_slots[course_id].add((semester, best_candidate['SlotID']))
        
        # Save all candidates before ML scoring
        all_candidates_df = pd.DataFrame(all_candidates)
        candidates_before_file = f"candidates_before_scoring_{model_name.lower()}.csv"
        all_candidates_df.to_csv(candidates_before_file, index=False)
        logger.info(f"Saved candidates before ML scoring to {candidates_before_file}")
        
        if progress_callback:
            progress_callback(f"Built timetable with {len(timetable)} scheduled classes for {len(scheduled_course_ids)} courses")
        
        timetable_df = pd.DataFrame(timetable)
        logger.info(f"Built timetable with {len(timetable_df)} scheduled classes for {len(scheduled_course_ids)} courses")
        
        # Save candidates after ML scoring with selected model score
        if len(all_candidates_df) > 0:
            scored_all_df, _ = ml_predictor.predict_slot_suitability(all_candidates_df, model_name, threshold=0.0)
            candidates_after_file = f"candidates_after_scoring_{model_name.lower()}.csv"
            scored_all_df.to_csv(candidates_after_file, index=False)
            logger.info(f"Saved candidates after ML scoring to {candidates_after_file}")
        
        # Try to schedule remaining unscheduled courses without ML filtering (if fallback enabled)
        if use_fallback:
            unscheduled_courses = self.courses_df[~self.courses_df['CourseID'].isin(scheduled_course_ids)]
            
            if len(unscheduled_courses) > 0:
                logger.info(f"Attempting to schedule {len(unscheduled_courses)} remaining courses without ML filtering...")
                
                for _, course in unscheduled_courses.iterrows():
                    course_id = course['CourseID']
                    level = course['LevelText']
                    semester = course['Semester']
                    level_group = course['LevelGroupID']
                    
                    if course_id in scheduled_course_ids:
                        continue
                    
                    # Generate candidates for this course with current schedule state
                    course_candidates = self._generate_candidates_for_course_incremental(
                        course_id, level, semester, level_group
                    )
                    
                    if len(course_candidates) == 0:
                        continue
                    
                    # Just pick the first valid candidate (no ML filtering)
                    for candidate in course_candidates:
                        if self._check_constraints_for_row(candidate, timetable):
                            timetable.append(candidate)
                            scheduled_course_ids.add(course_id)
                            
                            # Update schedule state
                            self.lecturer_schedule.add((candidate['LecturerID'], semester, candidate['SlotID']))
                            self.venue_schedule.add((candidate['VenueID'], semester, candidate['SlotID']))
                            if candidate['LevelAlreadyBooked'] == 0:
                                self.level_schedule.add((level, semester, candidate['SlotID']))
                            self.course_slots[course_id].add((semester, candidate['SlotID']))
                            break
        
        # Create result
        unscheduled_courses = self.courses_df[~self.courses_df['CourseID'].isin(scheduled_course_ids)]
        
        failures = []
        for _, course in unscheduled_courses.iterrows():
            failures.append(SchedulingFailure(
                course_id=course['CourseID'],
                course_code=course['CourseCode'],
                reason="Not scheduled by ML pipeline",
                constraint_violated=ConstraintType.WEEKLY_LIMIT
            ))
        
        result = ScheduleResult(
            timetable=timetable_df,
            scheduled_courses=scheduled_course_ids,
            unscheduled_courses=failures,
            conflicts_encountered={},
            score=len(timetable_df),
            debug_logs=[]
        )
        
        return result
    
    def format_timetable_view(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace IDs with human-readable fields and create pivot table view
        Following notebook procedure:
        """
        if timetable_df.empty:
            return pd.DataFrame()
        
        view_df = timetable_df.copy()
        
        # Merge with courses for CourseCode (if not already present)
        if 'CourseCode' not in view_df.columns:
            view_df = view_df.merge(
                self.courses_df[['CourseID', 'CourseCode']],
                on='CourseID',
                how='left'
            )
        
        # Merge with lecturers for LecturerName
        view_df = view_df.merge(
            self.lecturers_df[['LecturerID', 'LecturerName']],
            on='LecturerID',
            how='left'
        )
        
        # Merge with venues for VenueName
        view_df = view_df.merge(
            self.venues_df[['VenueID', 'VenueName']],
            on='VenueID',
            how='left'
        )
        
        # Merge with slots for Day and TimeRange
        view_df = view_df.merge(
            self.slots_df[['SlotID', 'Day', 'TimeRange']],
            on='SlotID',
            how='left'
        )
        
        # Keep readable columns
        readable_cols = [
            'CourseCode', 'LecturerName', 'LevelText', 'Semester',
            'Day', 'TimeRange', 'VenueName'
        ]
        
        # Only select columns that exist
        available_cols = [col for col in readable_cols if col in view_df.columns]
        view_df = view_df[available_cols]
        
        # Sort by day then time
        if 'Day' in view_df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            view_df['Day'] = pd.Categorical(view_df['Day'], categories=day_order, ordered=True)
            view_df = view_df.sort_values(['Day', 'TimeRange'] if 'TimeRange' in view_df.columns else ['Day'])
        
        logger.info(f"Formatted timetable view with {len(view_df)} rows")
        
        return view_df

    def format_timetable_grid(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proper timetable grid format using pivot table
        Following notebook procedure:
        - Index: Day, VenueName
        - Columns: TimeRange
        - Values: CourseCode
        """
        if len(timetable_df) == 0:
            return pd.DataFrame()
        
        # First get the readable view
        view_df = self.format_timetable_view(timetable_df)
        
        # Create pivot table for timetable grid
        try:
            timetable_grid = view_df.pivot_table(
                index=["Day", "VenueName"],
                columns="TimeRange",
                values="CourseCode",
                aggfunc="first"
            )
            
            logger.info(f"Created timetable grid with shape {timetable_grid.shape}")
            
            return timetable_grid
        except Exception as e:
            logger.error(f"Error creating timetable grid: {e}")
            # Fallback to readable view
            return view_df

    def _check_constraints_for_row(self, row, timetable: List) -> bool:
        """
        Check hard constraints for a candidate row against current timetable
        Only checks actual conflicts during timetable building
        """
        for r in timetable:
            # Lecturer conflict
            if r['SlotID'] == row['SlotID'] and r['LecturerID'] == row['LecturerID']:
                return False
            # Venue conflict
            if r['SlotID'] == row['SlotID'] and r['VenueID'] == row['VenueID']:
                return False
            # Level conflict
            if r['SlotID'] == row['SlotID'] and r['LevelText'] == row['LevelText']:
                return False
        
        return True
    
    def _update_schedule_state_from_timetable(self, timetable_df: pd.DataFrame) -> None:
        """Update schedule state from built timetable"""
        for _, row in timetable_df.iterrows():
            self.lecturer_schedule.add((row['LecturerID'], row['Semester'], row['SlotID']))
            self.venue_schedule.add((row['VenueID'], row['Semester'], row['SlotID']))
            self.level_schedule.add((row['LevelText'], row['Semester'], row['SlotID']))
            self.course_slots[row['CourseID']].add((row['Semester'], row['SlotID']))
            self.course_session_count[row['CourseID']] = self.course_session_count.get(row['CourseID'], 0) + 1

    def find_valid_assignment(
        self,
        course_id: int,
        course_code: str,
        level: int,
        semester: str,
        level_group: int,
        attempts: int = 100,
        ml_predictor=None,
        model_name='RandomForest'
    ) -> Optional[Dict]:
        """
        Find a valid slot/venue/lecturer assignment for a course
        Uses systematic search with fallback strategies
        Optionally uses ML to rank candidates
        """
        current_sessions = self.course_session_count[course_id]
        if current_sessions >= self.max_sessions_per_course:
            return None
        
        # Get qualified lecturers
        qualified_lecturers = self.get_qualified_lecturers(level)
        if not qualified_lecturers:
            logger.warning(f"No qualified lecturers for course {course_code} (level {level})")
            return None
        
        # Get suitable venues
        suitable_venues = [
            vid for vid, cap in self.venue_capacity.items()
            if cap >= self.level_sizes[level]
        ]
        if not suitable_venues:
            logger.warning(f"No suitable venues for course {course_code} (requires {self.level_sizes[level]} capacity)")
            return None
        
        # Generate all candidates
        candidates = []
        all_slots = self.slots_df.SlotID.tolist()
        slot_utilization = self.get_slot_utilization(semester)  # Get current slot usage
        
        for slot_id in all_slots:
            # Check if course already has this slot
            if (semester, slot_id) in self.course_slots[course_id]:
                continue
            
            for venue_id in suitable_venues:
                for lecturer_id in qualified_lecturers:
                    lecturer_conflict = 1 if self.check_lecturer_conflict(lecturer_id, semester, slot_id) else 0
                    venue_conflict = 1 if self.check_venue_conflict(venue_id, semester, slot_id) else 0
                    level_conflict = 1 if self.check_level_conflict(level, semester, slot_id) else 0
                    venue_capacity_ok = 1 if self.check_venue_capacity(venue_id, level) else 0
                    
                    candidate = {
                        'CourseID': course_id,
                        'CourseCode': course_code,
                        'LecturerID': lecturer_id,
                        'LevelText': level,
                        'LevelGroupID': level_group,
                        'Semester': semester,
                        'SlotID': slot_id,
                        'VenueID': venue_id,
                        'LecturerQualified': 1,
                        'VenueCapacitySuitable': venue_capacity_ok,
                        'LecturerAlreadyBooked': lecturer_conflict,
                        'VenueAlreadyBooked': venue_conflict,
                        'LevelAlreadyBooked': level_conflict,
                        '_slot_usage': slot_utilization.get(slot_id, 0)  # Internal field for sorting
                    }
                    candidates.append(candidate)
        
        # Filter to only conflict-free candidates first
        valid_candidates = [
            c for c in candidates
            if c['LecturerAlreadyBooked'] == 0
            and c['VenueAlreadyBooked'] == 0
            and c['VenueCapacitySuitable'] == 1
        ]
        
        # Sort valid candidates by slot usage (prefer less-used slots for load balancing)
        valid_candidates.sort(key=lambda c: c['_slot_usage'])
        
        # If ML predictor is available, use it to rank candidates
        if ml_predictor and len(valid_candidates) > 0:
            candidates_df = pd.DataFrame(valid_candidates)
            # Remove internal field before passing to ML
            candidates_df = candidates_df.drop(columns=['_slot_usage'])
            try:
                ranked_df = ml_predictor.rank_candidates(candidates_df, model_name=model_name)
                # Try candidates in ML-ranked order
                for _, row in ranked_df.head(attempts).iterrows():
                    if row['LevelAlreadyBooked'] == 0:
                        return row.to_dict()
                    # If no conflict-free candidates, try with level conflict
                    elif len([c for c in valid_candidates if c['LevelAlreadyBooked'] == 0]) == 0:
                        return row.to_dict()
            except Exception as e:
                logger.warning(f"ML ranking failed: {e}, falling back to systematic search")
        
        # Fallback to systematic search without ML
        # Strategy 1: Try candidates without level conflict first, prioritizing less-used slots
        non_conflict_candidates = [c for c in valid_candidates if c['LevelAlreadyBooked'] == 0]
        non_conflict_candidates.sort(key=lambda c: c['_slot_usage'])
        for candidate in non_conflict_candidates:
            result_dict = candidate.copy()
            result_dict.pop('_slot_usage', None)  # Remove internal field before returning
            return result_dict
        
        # Strategy 2: If no conflict-free candidates, allow level conflict but prefer less-used slots
        valid_candidates.sort(key=lambda c: c['_slot_usage'])
        if valid_candidates:
            result_dict = valid_candidates[0].copy()
            result_dict.pop('_slot_usage', None)  # Remove internal field before returning
            return result_dict
        
        # Strategy 3: Try all candidates including those with conflicts
        all_candidates = sorted(candidates, key=lambda c: c['_slot_usage'])
        for candidate in all_candidates:
            if candidate['LecturerAlreadyBooked'] == 0 and candidate['VenueAlreadyBooked'] == 0:
                result_dict = candidate.copy()
                result_dict.pop('_slot_usage', None)  # Remove internal field before returning
                return result_dict
        
        return None

    def schedule_courses(self, max_retries: int = 3, ml_predictor=None, model_name='RandomForest') -> ScheduleResult:
        """
        Schedule all courses using constraint satisfaction approach
        Optionally uses ML to guide assignment selection
        """
        logger.info("Starting course scheduling...")
        if ml_predictor:
            logger.info(f"Using ML-guided scheduling with {model_name} model")
        
        self.reset_schedule_state()
        
        timetable_rows = []
        failures = []
        conflicts_counter = defaultdict(int)
        debug_logs = []
        
        # Sort courses by level (higher levels first - they have fewer students)
        courses_to_schedule = self.courses_df.sort_values('LevelText', ascending=False).to_dict('records')
        
        total_courses = len(courses_to_schedule)
        scheduled_count = 0
        
        for course in courses_to_schedule:
            course_id = course['CourseID']
            course_code = course['CourseCode']
            level = course['LevelText']
            semester = course['Semester']
            level_group = course['LevelGroupID']
            
            # Try to schedule both sessions for this course
            for session in range(self.max_sessions_per_course):
                assignment = self.find_valid_assignment(
                    course_id, course_code, level, semester, level_group,
                    ml_predictor=ml_predictor, model_name=model_name
                )
                
                if assignment:
                    # Add to schedule
                    self.lecturer_schedule.add(
                        (assignment['LecturerID'], semester, assignment['SlotID'])
                    )
                    self.venue_schedule.add(
                        (assignment['VenueID'], semester, assignment['SlotID'])
                    )
                    if assignment['LevelAlreadyBooked'] == 0:
                        self.level_schedule.add(
                            (level, semester, assignment['SlotID'])
                        )
                    self.course_slots[course_id].add((semester, assignment['SlotID']))
                    self.course_session_count[course_id] += 1
                    
                    timetable_rows.append(assignment)
                    
                    # Track conflicts for scoring
                    if assignment['LevelAlreadyBooked'] == 1:
                        conflicts_counter['level_conflicts'] += 1
                    
                    debug_logs.append(
                        f"Scheduled {course_code} session {session + 1}: "
                        f"Slot {assignment['SlotID']}, Venue {assignment['VenueID']}, "
                        f"Lecturer {assignment['LecturerID']}"
                    )
                else:
                    # Record failure
                    failure = SchedulingFailure(
                        course_id=course_id,
                        course_code=course_code,
                        reason=f"Could not find valid assignment for session {session + 1}",
                        constraint_violated=ConstraintType.VENUE_CONFLICT  # Default
                    )
                    failures.append(failure)
                    debug_logs.append(
                        f"FAILED {course_code} session {session + 1}: "
                        f"No valid slot/venue/lecturer combination available"
                    )
                    conflicts_counter['failed_assignments'] += 1
            
            if self.course_session_count[course_id] > 0:
                scheduled_count += 1
        
        # Create timetable DataFrame
        timetable_df = pd.DataFrame(timetable_rows)
        
        # Get scheduled course IDs
        scheduled_course_ids = set(self.course_session_count.keys())
        
        # Calculate score
        score = self._calculate_score(timetable_df, scheduled_course_ids, conflicts_counter)
        
        logger.info(f"Scheduling complete: {scheduled_count}/{total_courses} courses scheduled")
        logger.info(f"Score: {score:.2f}")
        
        result = ScheduleResult(
            timetable=timetable_df,
            scheduled_courses=scheduled_course_ids,
            unscheduled_courses=failures,
            conflicts_encountered=dict(conflicts_counter),
            score=score,
            debug_logs=debug_logs
        )
        
        return result

    def _calculate_score(
        self,
        timetable_df: pd.DataFrame,
        scheduled_courses: Set[int],
        conflicts: Dict[str, int]
    ) -> float:
        """Calculate quality score for the timetable"""
        if len(timetable_df) == 0:
            return 0.0
        
        total_courses = len(self.courses_df)
        coverage_score = len(scheduled_courses) / total_courses
        
        # Penalty for conflicts
        conflict_penalty = conflicts.get('level_conflicts', 0) * 0.01
        failure_penalty = conflicts.get('failed_assignments', 0) * 0.05
        
        # Lecturer load balance (lower is better)
        lecturer_loads = Counter([r['LecturerID'] for _, r in timetable_df.iterrows()])
        load_variance = np.var(list(lecturer_loads.values())) if lecturer_loads else 0
        load_penalty = min(load_variance / 100, 0.2)
        
        # Venue utilization
        venue_utilization = len(timetable_df) / (len(self.venues_df) * len(self.slots_df) * 2)
        utilization_bonus = min(venue_utilization, 0.1)
        
        score = coverage_score - conflict_penalty - failure_penalty - load_penalty + utilization_bonus
        return max(0, min(1, score))

    def resolve_conflicts(self, result: ScheduleResult) -> ScheduleResult:
        """
        Attempt to resolve conflicts and unscheduled courses
        Uses backtracking and alternative strategies
        """
        logger.info("Attempting to resolve conflicts...")
        
        if not result.unscheduled_courses:
            logger.info("No conflicts to resolve - all courses scheduled")
            return result
        
        # Retry scheduling with relaxed constraints
        unscheduled_ids = {f.course_id for f in result.unscheduled_courses}
        
        # Reset and try again with different ordering
        self.reset_schedule_state()
        
        # Keep already scheduled courses
        scheduled_rows = result.timetable.to_dict('records')
        for row in scheduled_rows:
            self.lecturer_schedule.add((row['LecturerID'], row['Semester'], row['SlotID']))
            self.venue_schedule.add((row['VenueID'], row['Semester'], row['SlotID']))
            if row['LevelAlreadyBooked'] == 0:
                self.level_schedule.add((row['LevelText'], row['Semester'], row['SlotID']))
            self.course_slots[row['CourseID']].add((row['Semester'], row['SlotID']))
            self.course_session_count[row['CourseID']] += 1
        
        # Try to schedule failed courses with priority
        failed_courses = self.courses_df[self.courses_df['CourseID'].isin(unscheduled_ids)]
        
        new_timetable_rows = scheduled_rows.copy()
        new_failures = []
        
        for _, course in failed_courses.iterrows():
            course_id = course['CourseID']
            course_code = course['CourseCode']
            level = course['LevelText']
            semester = course['Semester']
            level_group = course['LevelGroupID']
            
            # Try with extended attempts
            assignment = self.find_valid_assignment(
                course_id, course_code, level, semester, level_group, attempts=500
            )
            
            if assignment:
                self.lecturer_schedule.add(
                    (assignment['LecturerID'], semester, assignment['SlotID'])
                )
                self.venue_schedule.add(
                    (assignment['VenueID'], semester, assignment['SlotID'])
                )
                if assignment['LevelAlreadyBooked'] == 0:
                    self.level_schedule.add((level, semester, assignment['SlotID']))
                
                self.course_slots[course_id].add((semester, assignment['SlotID']))
                self.course_session_count[course_id] += 1
                
                new_timetable_rows.append(assignment)
            else:
                new_failures.append(SchedulingFailure(
                    course_id=course_id,
                    course_code=course_code,
                    reason="Could not resolve even with relaxed constraints",
                    constraint_violated=ConstraintType.VENUE_CONFLICT
                ))
        
        new_timetable_df = pd.DataFrame(new_timetable_rows)
        new_scheduled = set(self.course_session_count.keys())
        
        new_result = ScheduleResult(
            timetable=new_timetable_df,
            scheduled_courses=new_scheduled,
            unscheduled_courses=new_failures,
            conflicts_encountered=result.conflicts_encountered,
            score=self._calculate_score(new_timetable_df, new_scheduled, result.conflicts_encountered),
            debug_logs=result.debug_logs + ["Attempted conflict resolution"]
        )
        
        logger.info(f"Resolution complete: {len(new_scheduled)}/{len(self.courses_df)} courses scheduled")
        
        return new_result

    def evaluate_schedule(self, result: ScheduleResult) -> Dict:
        """
        Evaluate the schedule and return detailed metrics
        """
        logger.info("Evaluating schedule...")
        
        if len(result.timetable) == 0:
            return {
                'total_courses': len(self.courses_df),
                'scheduled_courses': 0,
                'unscheduled_courses': len(self.courses_df),
                'coverage': 0.0,
                'score': 0.0,
                'lecturer_load': {},
                'venue_utilization': {},
                'day_distribution': {},
                'conflicts': result.conflicts_encountered
            }
        
        # Lecturer load
        lecturer_load = Counter([r['LecturerID'] for _, r in result.timetable.iterrows()])
        
        # Venue utilization
        venue_usage = Counter([r['VenueID'] for _, r in result.timetable.iterrows()])
        
        # Day distribution
        day_dist = Counter([self.slot_day[r['SlotID']] for _, r in result.timetable.iterrows()])
        
        evaluation = {
            'total_courses': len(self.courses_df),
            'scheduled_courses': len(result.scheduled_courses),
            'unscheduled_courses': len(result.unscheduled_courses),
            'coverage': len(result.scheduled_courses) / len(self.courses_df),
            'score': result.score,
            'lecturer_load': dict(lecturer_load),
            'venue_utilization': dict(venue_usage),
            'day_distribution': dict(day_dist),
            'conflicts': result.conflicts_encountered,
            'total_slots_assigned': len(result.timetable)
        }
        
        return evaluation

    def generate_ml_dataset(self, result: ScheduleResult) -> pd.DataFrame:
        """
        Generate ML-ready dataset from scheduling results
        Uses column names from ML pipeline (PascalCase)
        """
        logger.info("Generating ML dataset...")
        
        ml_data = []
        
        for _, row in result.timetable.iterrows():
            ml_row = {
                'CourseID': row['CourseID'],
                'LecturerID': row['LecturerID'],
                'LevelText': row['LevelText'],
                'LevelGroupID': row['LevelGroupID'],
                'Semester': row['Semester'],
                'SlotID': row['SlotID'],
                'VenueID': row['VenueID'],
                'LecturerQualified': row['LecturerQualified'],
                'VenueCapacitySuitable': row['VenueCapacitySuitable'],
                'LecturerAlreadyBooked': row['LecturerAlreadyBooked'],
                'VenueAlreadyBooked': row['VenueAlreadyBooked'],
                'LevelAlreadyBooked': row['LevelAlreadyBooked'],
                'TargetSuitable': 1,  # This assignment was successful
                'ConflictsEncountered': result.conflicts_encountered.get('level_conflicts', 0)
            }
            ml_data.append(ml_row)
        
        ml_df = pd.DataFrame(ml_data)
        
        logger.info(f"Generated ML dataset with {len(ml_df)} rows")
        
        return ml_df

    def run(self, max_attempts: int = 5, ml_predictor=None, model_name='RandomForest', use_fallback=True, progress_callback=None) -> Tuple[ScheduleResult, Dict]:
        """
        Main execution method
        Runs scheduling with ML pipeline if predictor provided, otherwise uses basic CSP
        """
        logger.info("=== Starting Smart Timetable Scheduler ===")
        
        if progress_callback:
            progress_callback("Loading data files...")
        
        # Load and validate data
        self.load_data()
        validation_result = self.validate_inputs()
        if not validation_result:
            raise ValueError("Input validation failed - Check logs for specific data issues. Common issues: Missing columns in CSV files, Invalid lecturer IDs in courses, Level values must be 100/200/300/400/500, Missing or empty data files.")
        
        if progress_callback:
            progress_callback("Building lookup dictionaries...")
        
        self.build_lookups()
        
        # Use ML pipeline if predictor provided
        if ml_predictor:
            logger.info(f"Using ML pipeline with {model_name} model")
            if progress_callback:
                progress_callback(f"Running ML pipeline with {model_name} model...")
            result = self.schedule_with_ml_pipeline(ml_predictor, model_name, threshold=0.75, shuffle=True, use_fallback=use_fallback, progress_callback=progress_callback)
            evaluation = self.evaluate_schedule(result)
            return result, evaluation
        
        # Otherwise use basic CSP approach
        logger.info("Using basic CSP scheduling")
        best_result = None
        best_score = -1
        best_evaluation = None
        
        for attempt in range(max_attempts):
            logger.info(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")
            self.reset_schedule_state()
            
            result = self.schedule_courses(max_retries=3, ml_predictor=ml_predictor, model_name=model_name)
            evaluation = self.evaluate_schedule(result)
            
            if result.score > best_score:
                best_result = result
                best_score = result.score
                best_evaluation = evaluation
                logger.info(f"New best score: {best_score}")
        
        logger.info(f"\n=== SCHEDULING COMPLETE ===")
        logger.info(f"Best result: {best_score} courses scheduled")
        
        return best_result, best_evaluation

    def _generate_candidates_for_course_incremental(
        self,
        course_id: int,
        level: str,
        semester: str,
        level_group: int
    ) -> list:
        """
        Generate candidates for a single course with current schedule state
        Constraint features reflect actual schedule state (like notebook)
        """
        expected_students = self.level_sizes.get(level, 100)
        
        candidates = []
        
        # Get qualified lecturers for this level
        qualified_lecturers = [
            lid for lid, rank in self.lecturer_rank.items()
            if level in self.qualification_rules.get(rank, [])
        ]
        
        if not qualified_lecturers:
            return []
        
        # Try all slots
        for _, slot in self.slots_df.iterrows():
            slot_id = slot['SlotID']
            
            # Check if course already has this slot
            if (semester, slot_id) in self.course_slots[course_id]:
                continue
            
            # Try all venues
            for _, venue in self.venues_df.iterrows():
                venue_id = venue['VenueID']
                venue_capacity = self.venue_capacity.get(venue_id, 0)
                capacity_suitable = 1 if venue_capacity >= expected_students else 0
                
                # Try all qualified lecturers
                for lecturer_id in qualified_lecturers:
                    # Check constraints based on current schedule state
                    lecturer_conflict = 1 if (lecturer_id, semester, slot_id) in self.lecturer_schedule else 0
                    venue_conflict = 1 if (venue_id, semester, slot_id) in self.venue_schedule else 0
                    level_conflict = 1 if (level, semester, slot_id) in self.level_schedule else 0
                    
                    row = {
                        "CourseID": course_id,
                        "LecturerID": lecturer_id,
                        "LevelText": level,
                        "LevelGroupID": level_group,
                        "Semester": semester,
                        "SlotID": slot_id,
                        "VenueID": venue_id,
                        "LecturerQualified": 1,
                        "VenueCapacitySuitable": capacity_suitable,
                        "LecturerAlreadyBooked": lecturer_conflict,
                        "VenueAlreadyBooked": venue_conflict,
                        "LevelAlreadyBooked": level_conflict
                    }
                    candidates.append(row)
        
        return candidates


def main():
    """Main entry point for testing"""
    scheduler = TimetableScheduler()
    
    try:
        result, evaluation = scheduler.run(max_attempts=5)
        
        print("\n" + "="*50)
        print("SCHEDULE EVALUATION")
        print("="*50)
        for key, value in evaluation.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*50)
        print("UNSCHEDULED COURSES")
        print("="*50)
        if result.unscheduled_courses:
            for failure in result.unscheduled_courses:
                print(f"{failure.course_code}: {failure.reason}")
        else:
            print("None - All courses scheduled successfully!")
        
        # Save timetable
        result.timetable.to_csv("final_timetable.csv", index=False)
    
    except AssertionError as e:
        logger.error(f"Coverage validation failed: {e}")
    except Exception as e:
        logger.error(f"Error during scheduling: {e}")


def main():
    """Main entry point for testing"""
    scheduler = TimetableScheduler()
    
    try:
        result, evaluation = scheduler.run(max_attempts=5)
        
        print("\n" + "="*50)
        print("SCHEDULE EVALUATION")
        print("="*50)
        for key, value in evaluation.items():
            print(f"{key}: {value}")
        
        print("\n" + "="*50)
        print("UNSCHEDULED COURSES")
        print("="*50)
        if result.unscheduled_courses:
            for failure in result.unscheduled_courses:
                print(f"{failure.course_code}: {failure.reason}")
        else:
            print("None - All courses scheduled successfully!")
        
        # Save timetable
        result.timetable.to_csv("final_timetable.csv", index=False)
    
    except AssertionError as e:
        logger.error(f"Coverage validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Scheduling failed: {e}")
        raise


if __name__ == "__main__":
    main()
