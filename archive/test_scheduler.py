"""
Comprehensive Test Script for Smart Timetable Scheduler
Tests both constraint-based and ML-guided scheduling
"""

import pandas as pd
import sys

from smart_timetable_scheduler import TimetableScheduler
from ml_timetable_predictor import TimetableMLPredictor


def test_basic_scheduler():
    """Test the basic constraint-based scheduler"""
    print("\n" + "="*70)
    print("TEST 1: Basic Constraint-Based Scheduler")
    print("="*70)
    
    scheduler = TimetableScheduler(data_dir=".")
    
    try:
        result, evaluation = scheduler.run(max_attempts=5)
        
        print("\n✓ Scheduling completed successfully")
        print(f"  - Total courses: {evaluation['total_courses']}")
        print(f"  - Scheduled courses: {evaluation['scheduled_courses']}")
        print(f"  - Coverage: {evaluation['coverage']:.2%}")
        print(f"  - Score: {evaluation['score']:.4f}")
        print(f"  - Total slots assigned: {evaluation['total_slots_assigned']}")
        
        # Check for unscheduled courses
        if result.unscheduled_courses:
            print(f"\n✗ FAILED: {len(result.unscheduled_courses)} courses not scheduled")
            for failure in result.unscheduled_courses:
                print(f"  - {failure.course_code}: {failure.reason}")
            return False
        else:
            print("\n✓ All courses scheduled successfully")
        
        # Save results
        result.timetable.to_csv("test_basic_timetable.csv", index=False)
        print("  - Timetable saved to test_basic_timetable.csv")
        
        # Generate ML dataset
        ml_dataset = scheduler.generate_ml_dataset(result)
        if not ml_dataset.empty:
            ml_dataset.to_csv("test_ml_dataset.csv", index=False)
            print(f"  - ML dataset saved to test_ml_dataset.csv ({len(ml_dataset)} rows)")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Coverage validation failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_guided_scheduler():
    """Test the ML-guided scheduler"""
    print("\n" + "="*70)
    print("TEST 2: ML-Guided Scheduler")
    print("="*70)
    
    # First, train or load ML models
    print("\nStep 1: Preparing ML models...")
    predictor = TimetableMLPredictor()
    
    # Try to load existing models
    try:
        predictor.load_models(".")
        print("  - Loaded existing models")
    except:
        # Try to train new models
        print("  - No existing models found, training new models...")
        try:
            df = predictor.load_training_data("timetable_training_dataset.csv")
            results = predictor.train_models(df)
            predictor.save_models(".")
            print("  - Models trained and saved successfully")
        except FileNotFoundError:
            print("  - No training data found. Run basic scheduler first to generate dataset.")
            print("  - Skipping ML-guided test")
            return None
    
    # Run ML-guided scheduling
    print("\nStep 2: Running ML-guided scheduling...")
    scheduler = TimetableScheduler(data_dir=".")
    
    try:
        result, evaluation = scheduler.run(max_attempts=5, ml_predictor=predictor)
        
        print("\n✓ ML-guided scheduling completed successfully")
        print(f"  - Total courses: {evaluation['total_courses']}")
        print(f"  - Scheduled courses: {evaluation['scheduled_courses']}")
        print(f"  - Coverage: {evaluation['coverage']:.2%}")
        print(f"  - Score: {evaluation['score']:.4f}")
        print(f"  - Total slots assigned: {evaluation['total_slots_assigned']}")
        
        # Check for unscheduled courses
        if result.unscheduled_courses:
            print(f"\n✗ FAILED: {len(result.unscheduled_courses)} courses not scheduled")
            for failure in result.unscheduled_courses:
                print(f"  - {failure.course_code}: {failure.reason}")
            return False
        else:
            print("\n✓ All courses scheduled successfully")
        
        # Save results
        result.timetable.to_csv("test_ml_guided_timetable.csv", index=False)
        print("  - Timetable saved to test_ml_guided_timetable.csv")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Coverage validation failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_schedulers():
    """Compare basic vs ML-guided scheduling results"""
    print("\n" + "="*70)
    print("TEST 3: Comparison Analysis")
    print("="*70)
    
    try:
        basic_df = pd.read_csv("test_basic_timetable.csv")
        ml_df = pd.read_csv("test_ml_guided_timetable.csv")
        
        print("\nBasic Scheduler:")
        print(f"  - Total assignments: {len(basic_df)}")
        print(f"  - Unique courses: {basic_df['course_id'].nunique()}")
        
        print("\nML-Guided Scheduler:")
        print(f"  - Total assignments: {len(ml_df)}")
        print(f"  - Unique courses: {ml_df['course_id'].nunique()}")
        
        # Compare conflict counts
        basic_conflicts = basic_df['level_conflict'].sum()
        ml_conflicts = ml_df['level_conflict'].sum()
        
        print("\nConflict Comparison:")
        print(f"  - Basic scheduler level conflicts: {basic_conflicts}")
        print(f"  - ML-guided level conflicts: {ml_conflicts}")
        
        if ml_conflicts < basic_conflicts:
            print(f"  ✓ ML-guided scheduler reduced conflicts by {basic_conflicts - ml_conflicts}")
        elif ml_conflicts > basic_conflicts:
            print(f"  ✗ ML-guided scheduler increased conflicts by {ml_conflicts - basic_conflicts}")
        else:
            print(f"  = Both schedulers have same number of conflicts")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ Comparison failed: {e}")
        print("  Run both basic and ML-guided tests first")
        return False
    except Exception as e:
        print(f"\n✗ Comparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_constraints():
    """Validate that the timetable satisfies all hard constraints"""
    print("\n" + "="*70)
    print("TEST 4: Constraint Validation")
    print("="*70)
    
    try:
        # Load the latest timetable
        import os
        if os.path.exists("test_ml_guided_timetable.csv"):
            timetable = pd.read_csv("test_ml_guided_timetable.csv")
            print("  - Using ML-guided timetable")
        elif os.path.exists("test_basic_timetable.csv"):
            timetable = pd.read_csv("test_basic_timetable.csv")
            print("  - Using basic timetable")
        else:
            print("  ✗ No timetable found to validate")
            return False
        
        # Check lecturer conflicts
        lecturer_slots = {}
        lecturer_conflicts = 0
        for _, row in timetable.iterrows():
            key = (row['lecturer_id'], row['semester'], row['slot_id'])
            if key in lecturer_slots:
                lecturer_conflicts += 1
            lecturer_slots[key] = row['course_code']
        
        print(f"\nLecturer Conflicts: {lecturer_conflicts}")
        if lecturer_conflicts > 0:
            print("  ✗ FAILED: Lecturer double-booking detected")
            return False
        else:
            print("  ✓ PASSED: No lecturer double-booking")
        
        # Check venue conflicts
        venue_slots = {}
        venue_conflicts = 0
        for _, row in timetable.iterrows():
            key = (row['venue_id'], row['semester'], row['slot_id'])
            if key in venue_slots:
                venue_conflicts += 1
            venue_slots[key] = row['course_code']
        
        print(f"Venue Conflicts: {venue_conflicts}")
        if venue_conflicts > 0:
            print("  ✗ FAILED: Venue double-booking detected")
            return False
        else:
            print("  ✓ PASSED: No venue double-booking")
        
        # Check venue capacity
        venues = pd.read_csv("venues.csv")
        venue_cap = dict(zip(venues.VenueID, venues.Capacity))
        level_sizes = {100: 250, 200: 200, 300: 150, 400: 120, 500: 80}
        
        capacity_violations = 0
        for _, row in timetable.iterrows():
            required = level_sizes.get(row['level'], 0)
            available = venue_cap.get(row['venue_id'], 0)
            if available < required:
                capacity_violations += 1
        
        print(f"Capacity Violations: {capacity_violations}")
        if capacity_violations > 0:
            print("  ✗ FAILED: Venue capacity violations detected")
            return False
        else:
            print("  ✓ PASSED: All venue capacities satisfied")
        
        print("\n✓ All hard constraints satisfied")
        return True
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SMART TIMETABLE SCHEDULER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Basic scheduler
    results['basic'] = test_basic_scheduler()
    
    # Test 2: ML-guided scheduler
    results['ml_guided'] = test_ml_guided_scheduler()
    
    # Test 3: Comparison (if both tests passed)
    if results['basic'] and results['ml_guided']:
        results['comparison'] = compare_schedulers()
    else:
        results['comparison'] = None
    
    # Test 4: Constraint validation
    if results['basic'] or results['ml_guided']:
        results['validation'] = validate_constraints()
    else:
        results['validation'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    # Overall result
    all_passed = all(r for r in results.values() if r is not None)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
