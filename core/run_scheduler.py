"""
Simple Standalone Timetable Scheduler
No server required - just run this script to generate timetable
"""

import pandas as pd
from smart_timetable_scheduler import TimetableScheduler
from ml_timetable_predictor import TimetableMLPredictor

def print_banner():
    print("=" * 60)
    print("       SMART TIMETABLE SCHEDULER")
    print("=" * 60)
    print()

def get_user_choice():
    print("Select Scheduling Method:")
    print("1. Basic (Constraint-based only) - Fast, no ML")
    print("2. RandomForest (ML-guided) - Best accuracy (95.5%)")
    print("3. GradientBoosting (ML-guided) - Good accuracy (94.25%)")
    print("4. LogisticRegression (ML-guided) - Good accuracy (89%)")
    print()
    
    while True:
        choice = input("Enter choice (1, 2, 3, or 4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

def run_scheduler(choice, use_fallback=True, progress_callback=None):
    print("\n" + "=" * 60)
    print("INITIALIZING SCHEDULER...")
    print("=" * 60)
    
    # Load ML predictor if needed
    ml_predictor = None
    model_name = "Basic"
    
    if choice in [2, 3, 4]:
        ml_predictor = TimetableMLPredictor()
        ml_predictor.load_models()

        if choice == 2:
            print("Loading RandomForest model...")
            model_name = "RandomForest"
            if not ml_predictor.rf_model:
                print("[ERROR] RandomForest model requested but failed to load from ml-models")
                print("Falling back to Basic scheduler")
                choice = 1
                ml_predictor = None
                model_name = "Basic"
            else:
                print("[OK] RandomForest model loaded successfully")

        elif choice == 3:
            print("Loading GradientBoosting model...")
            model_name = "GradientBoosting"
            if not ml_predictor.gb_model:
                print("[ERROR] GradientBoosting model requested but failed to load from ml-models")
                print("Falling back to Basic scheduler")
                choice = 1
                ml_predictor = None
                model_name = "Basic"
            else:
                print("[OK] GradientBoosting model loaded successfully")

        elif choice == 4:
            print("Loading LogisticRegression model...")
            model_name = "LogisticRegression"
            if not ml_predictor.lr_model:
                print("[ERROR] LogisticRegression model requested but failed to load from ml-models")
                print("Falling back to Basic scheduler")
                choice = 1
                ml_predictor = None
                model_name = "Basic"
            else:
                print("[OK] LogisticRegression model loaded successfully")
    
    print(f"\nSelected scheduler choice={choice}, model_name={model_name}, use_fallback={use_fallback}")
    
    # Use fallback parameter (skip input prompt for web app)
    if ml_predictor:
        print(f"Fallback: {'Enabled' if use_fallback else 'Disabled'}")
    
    print("\n" + "=" * 60)
    print("GENERATING TIMETABLE...")
    print("=" * 60)
    
    # Run scheduler
    scheduler = TimetableScheduler()
    result, evaluation = scheduler.run(max_attempts=5, ml_predictor=ml_predictor, model_name=model_name, use_fallback=use_fallback, progress_callback=progress_callback)
    
    return scheduler, result, evaluation, model_name

def display_results(result, evaluation, model_name):
    print("\n" + "=" * 60)
    print("SCHEDULING RESULTS")
    print("=" * 60)
    print(f"Model Used: {model_name}")
    print(f"Total Courses: {evaluation['total_courses']}")
    print(f"Scheduled: {evaluation['scheduled_courses']}")
    print(f"Unscheduled: {evaluation['unscheduled_courses']}")
    print(f"Coverage: {evaluation['coverage']:.2%}")
    print(f"Quality Score: {evaluation['score']:.4f}")
    print(f"Total Slots: {evaluation['total_slots_assigned']}")
    
    if result.unscheduled_courses:
        print("\n" + "-" * 60)
        print("UNSCHEDULED COURSES:")
        for failure in result.unscheduled_courses:
            print(f"  - {failure.course_code}: {failure.reason}")
    else:
        print("\n[OK] All courses scheduled successfully!")
    
    print("\n" + "=" * 60)
    print("LECTURER LOAD:")
    print("=" * 60)
    for lecturer_id, load in sorted(evaluation['lecturer_load'].items()):
        print(f"  Lecturer {lecturer_id}: {load} classes")
    
    print("\n" + "=" * 60)
    print("VENUE UTILIZATION:")
    print("=" * 60)
    for venue_id, usage in sorted(evaluation['venue_utilization'].items()):
        print(f"  Venue {venue_id}: {usage} classes")
    
    print("\n" + "=" * 60)
    print("DAY DISTRIBUTION:")
    print("=" * 60)
    for day, count in evaluation['day_distribution'].items():
        print(f"  {day}: {count} classes")

def save_results(scheduler, result, evaluation, model_name):
    """Save timetable and ML dataset to CSV files"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS...")
    print("=" * 60)
    
    # Save timetable with human-readable fields (list format)
    view_df = scheduler.format_timetable_view(result.timetable)
    timetable_file = f"timetable_{model_name.lower()}.csv"
    view_df.to_csv(timetable_file, index=False)
    print(f"[OK] Timetable list saved to: {timetable_file}")
    
    # Save timetable grid format (pivot table)
    grid_df = scheduler.format_timetable_grid(result.timetable)
    grid_file = f"timetable_grid_{model_name.lower()}.csv"
    grid_df.to_csv(grid_file)
    print(f"[OK] Timetable grid saved to: {grid_file}")
    
    # Save ML dataset
    ml_dataset = scheduler.generate_ml_dataset(result)
    dataset_file = f"dataset_{model_name.lower()}.csv"
    ml_dataset.to_csv(dataset_file, index=False)
    print(f"[OK] ML dataset saved to: {dataset_file}")

def generate_html_report(result, evaluation, model_name, scheduler):
    """Generate an HTML report for viewing in browser"""
    # Get human-readable timetable view
    view_df = scheduler.format_timetable_view(result.timetable)
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Timetable Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        th {{
            background: #f8f9fa;
            color: #333;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .success {{
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        h2 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Timetable Report</h1>
        
        <div class="success">
            <strong>[OK] Generated using: {model_name}</strong>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{evaluation['total_courses']}</div>
                <div class="stat-label">Total Courses</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{evaluation['scheduled_courses']}</div>
                <div class="stat-label">Scheduled</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{evaluation['coverage']:.1%}</div>
                <div class="stat-label">Coverage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{evaluation['score']:.3f}</div>
                <div class="stat-label">Quality Score</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📅 Timetable</h2>
            <table>
                <thead>
                    <tr>
                        <th>Course Code</th>
                        <th>Lecturer</th>
                        <th>Level</th>
                        <th>Semester</th>
                        <th>Day</th>
                        <th>Time</th>
                        <th>Venue</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add timetable rows with human-readable fields
    for _, row in view_df.iterrows():
        html += f"""
                    <tr>
                        <td>{row['CourseCode']}</td>
                        <td>{row['LecturerName']}</td>
                        <td>{row['LevelText']}</td>
                        <td>{row['Semester']}</td>
                        <td>{row['Day']}</td>
                        <td>{row['TimeRange']}</td>
                        <td>{row['VenueName']}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    html_file = f"timetable_{model_name.lower()}.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[OK] HTML report saved to: {html_file}")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f'file:///{html_file}')
        print("[OK] Report opened in browser")
    except:
        print("  (Could not auto-open browser)")

def main():
    print_banner()
    
    # Get user choice
    choice = get_user_choice()
    
    # Run scheduler
    scheduler, result, evaluation, model_name = run_scheduler(choice)
    
    # Display results
    display_results(result, evaluation, model_name)
    
    # Save results
    save_results(scheduler, result, evaluation, model_name)
    
    # Generate HTML report
    print("\n" + "=" * 60)
    print("GENERATING HTML REPORT...")
    print("=" * 60)
    generate_html_report(result, evaluation, model_name, scheduler)
    
    print("\n" + "=" * 60)
    print("[OK] DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
