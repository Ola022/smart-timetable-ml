"""
Enhanced Timetable Pivot Table Views
Restructured to follow web_app format and provide multiple visualization options
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from smart_timetable_scheduler import TimetableScheduler


class TimetablePivotViews:
    """
    Provides enhanced pivot table views for timetable display
    Following the format and style of web_app.py
    """
    
    def __init__(self, scheduler: TimetableScheduler):
        """
        Initialize with a scheduler instance to access data
        
        Args:
            scheduler: TimetableScheduler instance with loaded data
        """
        self.scheduler = scheduler
    
    def basic_pivot(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic pivot: Day & Venue as index, TimeRange as columns, CourseCode as values
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Pivot table with CourseCode values
        """
        view_df = self._prepare_view_df(timetable_df)
        
        pivot = view_df.pivot_table(
            index=["Day", "VenueName"],
            columns="TimeRange",
            values="CourseCode",
            aggfunc="first",
            sort=False
        )
        
        # Sort by day order
        pivot = self._sort_by_day_order(pivot)
        
        return pivot
    
    def enhanced_pivot_with_lecturer(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced pivot showing CourseCode + Lecturer info
        Format: "COURSE\n(Lecturer)"
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Pivot table with enhanced course information
        """
        view_df = self._prepare_view_df(timetable_df)
        
        # Create combined display value
        view_df['CourseWithLecturer'] = (
            view_df['CourseCode'] + '\n(' + 
            view_df['LecturerName'].str.split().str[-1] + ')'  # Last name only
        )
        
        pivot = view_df.pivot_table(
            index=["Day", "VenueName"],
            columns="TimeRange",
            values="CourseWithLecturer",
            aggfunc="first",
            sort=False
        )
        
        pivot = self._sort_by_day_order(pivot)
        
        return pivot
    
    def detailed_pivot_with_level(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detailed pivot showing CourseCode + Level + Lecturer
        Format: "COURSE-LVL\n(Lecturer)"
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Pivot table with detailed course information
        """
        view_df = self._prepare_view_df(timetable_df)
        
        # Create detailed display value
        view_df['CourseDetailed'] = (
            view_df['CourseCode'] + '-' + view_df['LevelText'].astype(str) + '\n(' + 
            view_df['LecturerName'].str.split().str[-1] + ')'
        )
        
        pivot = view_df.pivot_table(
            index=["Day", "VenueName"],
            columns="TimeRange",
            values="CourseDetailed",
            aggfunc="first",
            sort=False
        )
        
        pivot = self._sort_by_day_order(pivot)
        
        return pivot
    
    def venue_centric_pivot(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Venue-centric pivot: Venue as index, Day-TimeRange as multi-level columns
        Better for venue manager view
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Pivot table organized by venue
        """
        view_df = self._prepare_view_df(timetable_df)
        
        pivot = view_df.pivot_table(
            index="VenueName",
            columns=["Day", "TimeRange"],
            values="CourseCode",
            aggfunc="first",
            sort=False
        )
        
        return pivot
    
    def day_lecturer_pivot(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Day-Lecturer-centric pivot: Day & Lecturer as index, TimeRange as columns
        Better for lecturer schedule view
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Pivot table organized by day and lecturer
        """
        view_df = self._prepare_view_df(timetable_df)
        
        pivot = view_df.pivot_table(
            index=["Day", "LecturerName"],
            columns="TimeRange",
            values="CourseCode",
            aggfunc="first",
            sort=False
        )
        
        pivot = self._sort_by_day_order(pivot)
        
        return pivot
    
    def time_slot_pivot(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-slot view: TimeRange as index, Day as columns
        Better for conflict detection
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Pivot table organized by time slot
        """
        view_df = self._prepare_view_df(timetable_df)
        
        # Count courses per time-day combination
        pivot = view_df.pivot_table(
            index="TimeRange",
            columns="Day",
            values="CourseCode",
            aggfunc="count",
            sort=False
        )
        
        return pivot
    
    def summary_statistics(self, timetable_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics about the timetable
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Dictionary with summary statistics
        """
        view_df = self._prepare_view_df(timetable_df)
        
        stats = {
            'total_courses': len(view_df),
            'unique_days': view_df['Day'].nunique(),
            'unique_venues': view_df['VenueName'].nunique(),
            'unique_lecturers': view_df['LecturerName'].nunique(),
            'unique_timeslots': view_df['TimeRange'].nunique(),
            'courses_per_day': view_df.groupby('Day').size().to_dict(),
            'courses_per_venue': view_df.groupby('VenueName').size().to_dict(),
            'courses_per_lecturer': view_df.groupby('LecturerName').size().to_dict(),
            'courses_per_timeslot': view_df.groupby('TimeRange').size().to_dict(),
        }
        
        return stats
    
    def export_for_display(self, pivot_df: pd.DataFrame, fill_value: str = "---") -> pd.DataFrame:
        """
        Prepare pivot table for display (replace NaN with readable values)
        
        Args:
            pivot_df: Pivot table dataframe
            fill_value: Value to fill empty cells
            
        Returns:
            Pivot table ready for display
        """
        return pivot_df.fillna(fill_value)
    
    def export_for_csv(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Export timetable in readable CSV format
        Following the format from format_timetable_view
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Formatted dataframe for CSV export
        """
        return self.scheduler.format_timetable_view(timetable_df)
    
    # ==================== Helper Methods ====================
    
    def _prepare_view_df(self, timetable_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare view dataframe by merging all necessary data
        Follows the format_timetable_view approach
        
        Args:
            timetable_df: Raw timetable dataframe
            
        Returns:
            Prepared dataframe with all necessary columns
        """
        if timetable_df.empty:
            return pd.DataFrame()
        
        view_df = timetable_df.copy()
        
        # Merge with courses
        if 'CourseCode' not in view_df.columns:
            view_df = view_df.merge(
                self.scheduler.courses_df[['CourseID', 'CourseCode', 'LevelText']],
                on='CourseID',
                how='left'
            )
        
        # Merge with lecturers
        view_df = view_df.merge(
            self.scheduler.lecturers_df[['LecturerID', 'LecturerName']],
            on='LecturerID',
            how='left'
        )
        
        # Merge with venues
        view_df = view_df.merge(
            self.scheduler.venues_df[['VenueID', 'VenueName']],
            on='VenueID',
            how='left'
        )
        
        # Merge with slots
        view_df = view_df.merge(
            self.scheduler.slots_df[['SlotID', 'Day', 'TimeRange']],
            on='SlotID',
            how='left'
        )
        
        # Add Semester if not present
        if 'Semester' not in view_df.columns:
            view_df = view_df.merge(
                self.scheduler.courses_df[['CourseID', 'Semester']],
                on='CourseID',
                how='left'
            )
        
        return view_df
    
    def _sort_by_day_order(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort pivot table by standard day order
        
        Args:
            pivot_df: Pivot table to sort
            
        Returns:
            Sorted pivot table
        """
        if pivot_df.empty:
            return pivot_df
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        if isinstance(pivot_df.index, pd.MultiIndex):
            # For multi-index, we need to sort level 0 (Day)
            level_names = pivot_df.index.names
            if level_names[0] == 'Day':
                # Create a temporary dataframe to sort
                pivot_df = pivot_df.reset_index()
                pivot_df['Day'] = pd.Categorical(
                    pivot_df['Day'],
                    categories=day_order,
                    ordered=True
                )
                pivot_df = pivot_df.sort_values(['Day', pivot_df.columns[1]])
                pivot_df = pivot_df.set_index(level_names)
        
        return pivot_df
    
    def create_styled_output(self, pivot_df: pd.DataFrame, fill_value: str = "---") -> str:
        """
        Create a nicely formatted string representation of pivot table
        
        Args:
            pivot_df: Pivot table dataframe
            fill_value: Value to fill empty cells
            
        Returns:
            Formatted string representation
        """
        display_df = self.export_for_display(pivot_df, fill_value)
        return display_df.to_string()


# ==================== Usage Examples ====================

def demonstrate_pivot_views(scheduler, result):
    """
    Demonstrate different pivot table views
    
    Args:
        scheduler: TimetableScheduler instance
        result: ScheduleResult with timetable
    """
    pivot_views = TimetablePivotViews(scheduler)
    
    print("\n" + "="*80)
    print("TIMETABLE PIVOT VIEWS")
    print("="*80)
    
    # 1. Basic Pivot
    print("\n1. BASIC PIVOT (Day & Venue x TimeRange)")
    print("-" * 80)
    basic = pivot_views.basic_pivot(result.timetable)
    print(pivot_views.create_styled_output(basic))
    
    # 2. Enhanced Pivot with Lecturer
    print("\n2. ENHANCED PIVOT (with Lecturer)")
    print("-" * 80)
    enhanced = pivot_views.enhanced_pivot_with_lecturer(result.timetable)
    print(pivot_views.create_styled_output(enhanced))
    
    # 3. Detailed Pivot
    print("\n3. DETAILED PIVOT (with Level & Lecturer)")
    print("-" * 80)
    detailed = pivot_views.detailed_pivot_with_level(result.timetable)
    print(pivot_views.create_styled_output(detailed))
    
    # 4. Venue-Centric View
    print("\n4. VENUE-CENTRIC PIVOT")
    print("-" * 80)
    venue = pivot_views.venue_centric_pivot(result.timetable)
    print(venue.head(10).to_string())
    
    # 5. Lecturer-Centric View
    print("\n5. LECTURER-CENTRIC PIVOT")
    print("-" * 80)
    lecturer = pivot_views.day_lecturer_pivot(result.timetable)
    print(lecturer.head(10).to_string())
    
    # 6. Time Slot View
    print("\n6. TIME SLOT UTILIZATION")
    print("-" * 80)
    timeslot = pivot_views.time_slot_pivot(result.timetable)
    print(timeslot.to_string())
    
    # 7. Summary Statistics
    print("\n7. SUMMARY STATISTICS")
    print("-" * 80)
    stats = pivot_views.summary_statistics(result.timetable)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
