import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os
import io

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from smart_timetable_scheduler import TimetableScheduler
from ml_timetable_predictor import TimetableMLPredictor
from run_scheduler import run_scheduler, display_results, save_results

# Page configuration
st.set_page_config(
    page_title="SMART Timetable Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Generate Timetable'
if 'timetable_result' not in st.session_state:
    st.session_state.timetable_result = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'scheduler' not in st.session_state:
    st.session_state.scheduler = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# Sidebar
st.sidebar.header("📅 SMART Scheduler")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    options=["Generate Timetable", "View Timetable", "Manage Data", "View Data"],
    index=["Generate Timetable", "View Timetable", "Manage Data", "View Data"].index(st.session_state.page)
)
st.session_state.page = page

# Scheduling configuration (only show on Generate/View pages)
if page in ["Generate Timetable", "View Timetable"]:
    st.sidebar.subheader("Configuration")
    
    method = st.sidebar.radio(
        "Scheduling Method:",
        options=["Basic (Constraint-based)", "RandomForest (ML)", "GradientBoosting (ML)", "LogisticRegression (ML)"],
        index=1
    )
    
    method_map = {
        "Basic (Constraint-based)": 1,
        "RandomForest (ML)": 2,
        "GradientBoosting (ML)": 3,
        "LogisticRegression (ML)": 4
    }
    choice = method_map[method]
    
    # Fallback option for ML methods
    use_fallback = True
    if choice in [2, 3, 4]:
        use_fallback = st.sidebar.checkbox("Enable fallback for unscheduled courses", value=True)
        st.sidebar.info("Fallback schedules remaining courses without ML filtering")
    
    # Max attempts (only for Basic)
    max_attempts = 5
    if choice == 1:
        max_attempts = st.sidebar.slider("Max scheduling attempts", min_value=1, max_value=10, value=5)
    
    # Threshold for ML
    threshold = 0.75
    if choice in [2, 3, 4]:
        threshold = st.sidebar.slider("ML Threshold", min_value=0.5, max_value=0.95, value=0.75, step=0.05)

# Main content area
if page == "Generate Timetable":
    st.markdown('<div class="main-header">📅 SMART Timetable Scheduler</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("About")
        st.markdown("""
        This intelligent timetable scheduler uses machine learning to optimize course scheduling.
        
        **Features:**
        - ML-guided scheduling for optimal placement
        - Constraint satisfaction (lecturers, venues, time slots)
        - Conflict detection and resolution
        - Multiple scheduling algorithms to choose from
        - 100% course coverage with fallback option
        """)
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Courses", "106")
        st.metric("Total Lecturers", "31")
        st.metric("Total Venues", "5")
        st.metric("Total Slots", "30")
    
    st.markdown("---")
    st.info("👉 Navigate to 'View Timetable' to generate and view timetables")

elif page == "View Timetable":
    st.markdown('<div class="main-header">📋 View Timetable</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Configuration summary
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Method:** {method} \t
        **Fallback:** {'Enabled' if use_fallback else 'Disabled'}        
        """)
    with col2:
        if choice in [2, 3, 4]:
            # Get model name from method selection
            model_name_display = method.split('(')[0].strip()
            st.info(f"""
            **ML Threshold:** {threshold} \t
            **Algorithm:** {model_name_display}
            """)
        else:
            st.info(f"""
            **Algorithm:** Basic Constraint-based
            """)
    
    st.markdown("---")
    
    # Generate button
    run_button = st.button("🚀 Generate Timetable", type="primary", use_container_width=True)
    
    if run_button:
        # Create a status container for real-time progress
        progress_container = st.container()
        status_placeholder = progress_container.empty()
        
        try:
            # Progress callback function
            def progress_callback(message):
                status_placeholder.info(f"⏳ {message}")
            
            # Run scheduler with progress display
            progress_callback("Initializing scheduler...")
            
            scheduler, result, evaluation, model_name = run_scheduler(
                choice, 
                use_fallback=use_fallback,
                progress_callback=progress_callback
            )
            
            # Store in session state
            st.session_state.timetable_result = result
            st.session_state.evaluation = evaluation
            st.session_state.scheduler = scheduler
            st.session_state.model_name = model_name
            
            status_placeholder.success("✅ Timetable generated successfully!")
            
        except Exception as e:
            status_placeholder.error(f"❌ Error generating timetable: {str(e)}")
            st.exception(e)
    
    # Display results if available
    if st.session_state.timetable_result is not None:
        st.markdown("---")
        st.subheader("Scheduling Results")
        
        result = st.session_state.timetable_result
        evaluation = st.session_state.evaluation
        scheduler = st.session_state.scheduler
        model_name = st.session_state.model_name
        
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Courses", evaluation['total_courses'])
        with col2:
            st.metric("Scheduled", evaluation['scheduled_courses'])
        with col3:
            st.metric("Coverage", f"{evaluation['coverage']:.1%}")
        with col4:
            st.metric("Quality Score", f"{evaluation['score']:.2f}")
        
        # Algorithm info
        st.info(f"""
        **Algorithm Used:** {model_name}
        **Fallback:** {'Enabled' if use_fallback else 'Disabled'} if choice in [2,3,4] else 'N/A'
        """)
        
        # Unscheduled courses warning
        if result.unscheduled_courses:
            st.warning(f"⚠️ {len(result.unscheduled_courses)} courses could not be scheduled")
            with st.expander("View unscheduled courses"):
                for failure in result.unscheduled_courses:
                    st.write(f"- {failure.course_code}: {failure.reason}")
        else:
            st.success("✅ All courses scheduled successfully!")
        
        st.markdown("---")
        
        # Charts in tabs
        st.subheader("Analytics & Timetable")
        tab1, tab2 = st.tabs(["Timetable", "Analytics"])
        
        with tab1:
            # Filters
            st.subheader("Filters")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                search_term = st.text_input("Search (Course Code):", "", key="timetable_search")
            
            with col2:
                view_df = scheduler.format_timetable_view(result.timetable)
                if 'Semester' in view_df.columns:
                    semesters = ['All'] + sorted(view_df['Semester'].unique().tolist())
                    semester_filter = st.selectbox("Filter by Semester:", semesters)
                else:
                    semester_filter = 'All'
            
            with col3:
                if 'LecturerName' in view_df.columns:
                    lecturers = ['All'] + sorted(view_df['LecturerName'].unique().tolist())
                    lecturer_filter = st.selectbox("Filter by Lecturer:", lecturers)
                else:
                    lecturer_filter = 'All'
            
            with col4:
                if 'VenueName' in view_df.columns:
                    venues = ['All'] + sorted(view_df['VenueName'].unique().tolist())
                    venue_filter = st.selectbox("Filter by Venue:", venues)
                else:
                    venue_filter = 'All'
            
            # Apply filters
            filtered_df = view_df.copy()
            
            if search_term:
                filtered_df = filtered_df[filtered_df['CourseCode'].str.contains(search_term, case=False, na=False)]
            
            if semester_filter != 'All':
                filtered_df = filtered_df[filtered_df['Semester'] == semester_filter]
            
            if lecturer_filter != 'All':
                filtered_df = filtered_df[filtered_df['LecturerName'] == lecturer_filter]
            
            if venue_filter != 'All':
                filtered_df = filtered_df[filtered_df['VenueName'] == venue_filter]
            
            st.markdown("---")
            
            # Display filtered results
            st.subheader(f"Timetable ({len(filtered_df)} entries)")
            st.dataframe(filtered_df, use_container_width=True, height=500)
            
            # Download filtered
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered View",
                data=csv,
                file_name=f"timetable_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab2:
            # Load reference data for names
            lecturers_df = pd.read_csv('data_files/lecturers.csv')
            venues_df = pd.read_csv('data_files/venues.csv')
            slots_df = pd.read_csv('data_files/timeslots.csv')
            
            # Create ID to name mappings
            lecturer_id_to_name = dict(zip(lecturers_df['LecturerID'], lecturers_df['LecturerName']))
            venue_id_to_name = dict(zip(venues_df['VenueID'], venues_df['VenueName']))
            slot_id_to_name = dict(zip(slots_df['SlotID'], slots_df['TimeRange']))
            
            # Row 1: Lecturer Load (full width)
            st.subheader("Lecturer Load Distribution")
            lecturer_df = pd.DataFrame(
                list(evaluation['lecturer_load'].items()),
                columns=['Lecturer ID', 'Classes']
            )
            # Replace IDs with names
            lecturer_df['Lecturer Name'] = lecturer_df['Lecturer ID'].map(lecturer_id_to_name)
            lecturer_df = lecturer_df[['Lecturer Name', 'Classes']]
            st.bar_chart(lecturer_df.set_index('Lecturer Name'))
            
            # Row 2: Venue Utilization and Time Slot Distribution (2 columns)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Venue Utilization")
                venue_df = pd.DataFrame(
                    list(evaluation['venue_utilization'].items()),
                    columns=['Venue ID', 'Classes']
                )
                # Replace IDs with names
                venue_df['Venue Name'] = venue_df['Venue ID'].map(venue_id_to_name)
                venue_df = venue_df[['Venue Name', 'Classes']]
                st.bar_chart(venue_df.set_index('Venue Name'))
            
            with col2:
                st.subheader("Time Slot Distribution")
                day_df = pd.DataFrame(
                    list(evaluation['day_distribution'].items()),
                    columns=['Day', 'Classes']
                )
                st.bar_chart(day_df.set_index('Day'))
        
        st.markdown("---")
        
        # Download buttons
        st.subheader("Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = result.timetable.to_csv(index=False)
            st.download_button(
                label="📥 Download Timetable CSV",
                data=csv,
                file_name=f"timetable_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            view_df = scheduler.format_timetable_view(result.timetable)
            view_csv = view_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Readable CSV",
                data=view_csv,
                file_name=f"timetable_readable_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            html_content = f"""
            <html>
            <head>
                <title>Timetable - {model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Timetable - {model_name}</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Courses: {evaluation['total_courses']}</p>
                <p>Scheduled: {evaluation['scheduled_courses']}</p>
                <p>Coverage: {evaluation['coverage']:.1%}</p>
                <h2>Timetable</h2>
                {view_df.to_html(index=False)}
            </body>
            </html>
            """
            st.download_button(
                label="📥 Download HTML Report",
                data=html_content,
                file_name=f"timetable_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
        
        st.markdown("---")
        
        
elif page == "Manage Data":
    st.markdown('<div class="main-header">➕ Manage Data</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    data_type = st.radio("Select data type to manage:", ["Courses", "Lecturers", "Venues"], key="manage_data_type")
    
    if data_type == "Courses":
        st.subheader("Add New Course")
        
        # Load lecturers for dropdown
        lecturers_df = pd.read_csv('data_files/lecturers.csv')
        lecturer_options = lecturers_df[['LecturerID', 'LecturerName']].values.tolist()
        lecturer_dict = {f"{row[1]} (ID: {row[0]})": row[0] for row in lecturer_options}
        
        with st.form("add_course"):
            col1, col2 = st.columns(2)
            with col1:
                course_code = st.text_input("Course Code (e.g., CSC101):", key="course_code")
                level = st.text_input("Level (e.g., 100):", key="level")
                semester = st.selectbox("Semester", ["First", "Second"], key="course_semester")
            with col2:
                lecturer_name = st.selectbox("Lecturer", list(lecturer_dict.keys()), key="course_lecturer")
                department = st.text_input("Department (e.g., Computer Science):", key="department")
            
            submitted = st.form_submit_button("Add Course")
            if submitted and course_code and level:
                # Load existing courses
                courses_df = pd.read_csv('data_files/courses.csv')
                new_id = courses_df['CourseID'].max() + 1 if len(courses_df) > 0 else 1
                
                # Get lecturer ID from selection
                lecturer_id = lecturer_dict[lecturer_name]
                
                # Auto-generate LevelGroupID from level
                level_group_map = {
                    '100': 13, '200': 14, '300': 15, '400': 16, '500': 17
                }
                level_group = level_group_map.get(level, 13)
                
                new_course = pd.DataFrame([{
                    'CourseID': new_id,
                    'CourseCode': course_code,
                    'LecturerID': lecturer_id,
                    'LevelText': level,
                    'LevelGroupID': level_group,
                    'Department': department,
                    'Semester': semester
                }])
                courses_df = pd.concat([courses_df, new_course], ignore_index=True)
                courses_df.to_csv('data_files/courses.csv', index=False)
                st.success(f"✅ Course {course_code} added successfully!")
    
    elif data_type == "Lecturers":
        st.subheader("Add New Lecturer")
        with st.form("add_lecturer"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Lecturer Name:", key="lecturer_name")
                rank = st.selectbox("Rank", ["Professor", "Associate Professor", "Senior Lecturer", "Lecturer"], key="lecturer_rank")
            with col2:
                department = st.text_input("Department:", key="lecturer_department")
            
            submitted = st.form_submit_button("Add Lecturer")
            if submitted and name:
                lecturers_df = pd.read_csv('data_files/lecturers.csv')
                new_id = lecturers_df['LecturerID'].max() + 1 if len(lecturers_df) > 0 else 1
                new_lecturer = pd.DataFrame([{
                    'LecturerID': new_id,
                    'LecturerName': name,
                    'Rank': rank,
                    'Department': department
                }])
                lecturers_df = pd.concat([lecturers_df, new_lecturer], ignore_index=True)
                lecturers_df.to_csv('data_files/lecturers.csv', index=False)
                st.success(f"✅ Lecturer {name} added successfully!")
    
    elif data_type == "Venues":
        st.subheader("Add New Venue")
        with st.form("add_venue"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Venue Name:", key="venue_name")
                capacity = st.number_input("Capacity", min_value=1, value=50, key="venue_capacity")
            with col2:
                building = st.text_input("Building:", key="venue_building")
                room_type = st.selectbox("Room Type", ["Lecture Hall", "Lab", "Seminar Room"], key="venue_type")
            
            submitted = st.form_submit_button("Add Venue")
            if submitted and name:
                venues_df = pd.read_csv('data_files/venues.csv')
                new_id = venues_df['VenueID'].max() + 1 if len(venues_df) > 0 else 1
                new_venue = pd.DataFrame([{
                    'VenueID': new_id,
                    'VenueName': name,
                    'Capacity': capacity,
                    'Building': building,
                    'Type': room_type
                }])
                venues_df = pd.concat([venues_df, new_venue], ignore_index=True)
                venues_df.to_csv('data_files/venues.csv', index=False)
                st.success(f"✅ Venue {name} added successfully!")
    
    
elif page == "View Data":
    st.markdown('<div class="main-header">📊 View Data</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    data_type = st.radio("Select data to view:", ["Courses", "Lecturers", "Venues", "Time Slots"], key="view_data_type")
    
    # Search and filter
    search_term = st.text_input("Search:", "", key="view_data_search")
    
    if data_type == "Courses":
        df = pd.read_csv('data_files/courses.csv')
        
        # Merge with lecturers to show names
        lecturers_df = pd.read_csv('data_files/lecturers.csv')
        df = df.merge(lecturers_df[['LecturerID', 'LecturerName']], on='LecturerID', how='left')
        
        # Reorder columns to show LecturerName instead of LecturerID
        cols = ['CourseID', 'CourseCode', 'LecturerName', 'LevelText', 'LevelGroupID', 'Department', 'Semester']
        df = df[cols]
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        st.subheader(f"Courses ({len(df)} entries)")
        st.dataframe(df, use_container_width=True)
    
    elif data_type == "Lecturers":
        df = pd.read_csv('data_files/lecturers.csv')
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        st.subheader(f"Lecturers ({len(df)} entries)")
        st.dataframe(df, use_container_width=True)
    
    elif data_type == "Venues":
        df = pd.read_csv('data_files/venues.csv')
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        st.subheader(f"Venues ({len(df)} entries)")
        st.dataframe(df, use_container_width=True)
    
    elif data_type == "Time Slots":
        df = pd.read_csv('data_files/timeslots.csv')
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        st.subheader(f"Time Slots ({len(df)} entries)")
        st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SMART Timetable Scheduler v1.0 | ML-Guided Intelligent Scheduling</p>
</div>
""", unsafe_allow_html=True)

