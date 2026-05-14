import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os
import io
import html

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from smart_timetable_scheduler import TimetableScheduler
from ml_timetable_predictor import TimetableMLPredictor
from run_scheduler import run_scheduler, display_results, save_results


def dataframe_to_excel_xml(df):
    """Create an Excel-compatible XML workbook with wrapped multiline cells."""
    lines = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
        ' xmlns:o="urn:schemas-microsoft-com:office:office"',
        ' xmlns:x="urn:schemas-microsoft-com:office:excel"',
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">',
        '<Styles>',
        '<Style ss:ID="wrap"><Alignment ss:Vertical="Top" ss:WrapText="1"/></Style>',
        '<Style ss:ID="header"><Font ss:Bold="1"/><Alignment ss:Vertical="Top" ss:WrapText="1"/></Style>',
        '</Styles>',
        '<Worksheet ss:Name="Timetable">',
        '<Table>'
    ]

    lines.append('<Row>')
    for column in df.columns:
        value = html.escape(str(column), quote=True)
        lines.append(f'<Cell ss:StyleID="header"><Data ss:Type="String">{value}</Data></Cell>')
    lines.append('</Row>')

    for _, row in df.iterrows():
        lines.append('<Row>')
        for value in row:
            cell_value = '' if pd.isna(value) else html.escape(str(value), quote=True).replace('\n', '&#10;')
            lines.append(f'<Cell ss:StyleID="wrap"><Data ss:Type="String">{cell_value}</Data></Cell>')
        lines.append('</Row>')

    lines.extend([
        '</Table>',
        '</Worksheet>',
        '</Workbook>'
    ])
    return '\n'.join(lines).encode('utf-8')


# Page configuration
st.set_page_config(
    page_title="SMART Timetable Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit header and navbar
hide_streamlit_style = """
    <style>
    /* Hide main header */
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Hide sidebar navigation elements */
    .css-1lcbmhc {display: none;}
    .css-1q8dd3 {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
    st.session_state.page = 'Overview'
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
    options=["Overview", "View Timetable", "Manage Data", "View Data"],
    index=["Overview", "View Timetable", "Manage Data", "View Data"].index(st.session_state.page)
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
if page == "Overview":
    st.markdown('<div class="main-header">📅 SMART Timetable Scheduler</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Owner info
    st.info("""
    **Name:** Bawa Yusuff Ayodele
    **Matric Number:** 200211
    **Version:** v1.0 | ML-Guided Intelligent Scheduling
    """)
    
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
            
            # Create a pivot table from the filtered timetable view
            if len(filtered_df) > 0:
                filtered_df = filtered_df.copy()
                
                # Add venue info directly into the displayed course value
                filtered_df['CourseWithVenue'] = (
                    filtered_df['CourseCode'] + ' - ' + filtered_df['VenueName']
                )
                
                def time_range_sort_key(time_range):
                    start_time = str(time_range).split('-')[0].strip()
                    try:
                        return datetime.strptime(start_time, '%H:%M').time()
                    except ValueError:
                        return datetime.max.time()
                
                def join_lines(values):
                    values = [str(v) for v in values if pd.notna(v) and str(v).strip()]
                    return '\n'.join(values)
                
                timetable_pivot = filtered_df.pivot_table(
                    index='Day',
                    columns='TimeRange',
                    values='CourseWithVenue',
                    aggfunc=join_lines,
                    sort=False
                )
                
                # Keep a consistent day order
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                timetable_pivot = timetable_pivot.reindex(day_order).fillna('')
                sorted_time_ranges = sorted(timetable_pivot.columns, key=time_range_sort_key)
                timetable_pivot = timetable_pivot.reindex(columns=sorted_time_ranges)
                
                st.subheader(f"Timetable Pivot ({len(filtered_df)} entries)")
                
                # Convert to HTML with proper line break rendering
                display_pivot = timetable_pivot.copy()
                for column in display_pivot.columns:
                    display_pivot[column] = display_pivot[column].map(
                        lambda value: html.escape(str(value)).replace('\n', '<br>') if str(value).strip() else ''
                    )
                html_table = display_pivot.to_html(escape=False)
                st.markdown(html_table, unsafe_allow_html=True)
                
                download_data = dataframe_to_excel_xml(timetable_pivot.reset_index())
                download_file_name = f"timetable_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xls"
                download_mime = "application/vnd.ms-excel"
            else:
                st.warning("No timetable entries match the selected filters.")
                download_data = filtered_df.to_csv(index=False)
                download_file_name = f"timetable_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                download_mime = "text/csv"
            
            st.download_button(
                label="📥 Download Pivot View",
                data=download_data,
                file_name=download_file_name,
                mime=download_mime
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
                level = st.selectbox("Level:", options=['100', '200', '300', '400', '500', '600', '700', '800', '900'], 
                                       index=0, key="course_level")
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
        
        # Row selection
        event = st.dataframe(df, use_container_width=True, height=400, on_select="rerun", key="courses_df")
        selected_rows = event.selection.rows
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Edit Selected", key="edit_course"):
                if selected_rows:
                    # Clear any existing edit state first
                    st.session_state.edit_mode = True
                    st.session_state.edit_data_type = 'courses'
                    st.session_state.edit_row_index = selected_rows[0]
                    st.session_state.edit_data = df.iloc[selected_rows[0]].to_dict()
                    st.rerun()
                else:
                    st.info("Select a row to edit")
        
        with col2:
            if st.button("🗑️ Delete Selected", key="delete_course"):
                if selected_rows:
                    st.session_state.delete_mode = True
                    st.session_state.delete_data_type = 'courses'
                    st.session_state.delete_rows = selected_rows
                    st.session_state.delete_data = df.iloc[selected_rows]
                    st.rerun()
                else:
                    st.info("Select a row to delete")
        
        # Edit mode
        if st.session_state.get('edit_mode', False) and st.session_state.get('edit_data_type') == 'courses':
            st.subheader("Edit Course")
            edit_data = st.session_state.edit_data
            
            # Load lecturers for dropdown (outside form)
            lecturers_df = pd.read_csv('data_files/lecturers.csv')
            lecturer_options = lecturers_df[['LecturerID', 'LecturerName']].values.tolist()
            lecturer_dict = {f"{row[1]} (ID: {row[0]})": row[0] for row in lecturer_options}
            
            # Check if LecturerID exists in edit_data
            if 'LecturerID' in edit_data:
                current_lecturer = f"{edit_data['LecturerName']} (ID: {edit_data['LecturerID']})"
            else:
                # Fallback: find LecturerID from LecturerName
                matching_lecturer = lecturers_df[lecturers_df['LecturerName'] == edit_data['LecturerName']]
                if not matching_lecturer.empty:
                    lecturer_id = matching_lecturer.iloc[0]['LecturerID']
                    current_lecturer = f"{edit_data['LecturerName']} (ID: {lecturer_id})"
                else:
                    current_lecturer = list(lecturer_dict.keys())[0]  # Default to first lecturer
            
            with st.form("edit_course_form"):
                col1, col2 = st.columns(2)
                with col1:
                    course_code = st.text_input("Course Code:", value=edit_data['CourseCode'], key="edit_course_code")
                    # Level dropdown with validation
                    level_options = ['100', '200', '300', '400', '500', '600', '700', '800', '900']
                    current_level = str(edit_data['LevelText']) if isinstance(edit_data['LevelText'], (int, float)) else edit_data['LevelText']
                    level = st.selectbox("Level:", options=level_options, 
                                       index=level_options.index(current_level) if current_level in level_options else 0,
                                       key="edit_level")
                    semester = st.selectbox("Semester", ["First", "Second"], 
                                           index=0 if edit_data['Semester'] == 'First' else 1, 
                                           key="edit_semester")
                with col2:
                    lecturer_name = st.selectbox("Lecturer", list(lecturer_dict.keys()), 
                                                index=list(lecturer_dict.keys()).index(current_lecturer) if current_lecturer in lecturer_dict else 0,
                                                key="edit_lecturer")
                    department = st.text_input("Department:", value=edit_data['Department'], key="edit_department")
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    submitted = st.form_submit_button("💾 Save Changes", type="primary")
                with col_cancel:
                    cancelled = st.form_submit_button("❌ Cancel")
                
                if submitted:
                    # Update course
                    courses_df = pd.read_csv('data_files/courses.csv')
                    course_id = edit_data['CourseID']
                    lecturer_id = lecturer_dict[lecturer_name]
                    
                    # Auto-generate LevelGroupID from level
                    level_group_map = {
                        '100': 13, '200': 14, '300': 15, '400': 16, '500': 17
                    }
                    level_group = level_group_map.get(level, 13)
                    
                    # Convert level to proper type and ensure LevelText matches original dtype
                    try:
                        level_value = int(level)
                        # Check original dtype of LevelText column
                        original_dtype = courses_df['LevelText'].dtype
                        if pd.api.types.is_integer_dtype(original_dtype):
                            level_text = level_value  # Keep as integer
                        else:
                            level_text = str(level_value)  # Convert to string
                    except ValueError:
                        level_text = level  # Keep original string if not convertible
                    
                    # Update row with proper types
                    courses_df.loc[courses_df['CourseID'] == course_id, 'CourseCode'] = course_code
                    courses_df.loc[courses_df['CourseID'] == course_id, 'LecturerID'] = lecturer_id
                    courses_df.loc[courses_df['CourseID'] == course_id, 'LevelText'] = level_text
                    courses_df.loc[courses_df['CourseID'] == course_id, 'LevelGroupID'] = level_group
                    courses_df.loc[courses_df['CourseID'] == course_id, 'Department'] = department
                    courses_df.loc[courses_df['CourseID'] == course_id, 'Semester'] = semester
                    
                    courses_df.to_csv('data_files/courses.csv', index=False)
                    st.success("✅ Course updated successfully!")
                    st.session_state.edit_mode = False
                    st.rerun()
                
                if cancelled:
                    st.session_state.edit_mode = False
                    st.rerun()
        
        # Delete confirmation
        if st.session_state.get('delete_mode', False) and st.session_state.get('delete_data_type') == 'courses':
            delete_data = st.session_state.delete_data
            st.warning(f"⚠️ Are you sure you want to delete {len(delete_data)} course(s)?")
            
            # Show courses to be deleted
            st.dataframe(delete_data[['CourseCode', 'LecturerName', 'Department', 'Semester']], 
                        use_container_width=True, height=200)
            
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Yes, Delete", key="confirm_delete_course", type="primary"):
                    # Perform deletion
                    courses_df = pd.read_csv('data_files/courses.csv')
                    course_ids_to_delete = delete_data['CourseID'].tolist()
                    courses_df = courses_df[~courses_df['CourseID'].isin(course_ids_to_delete)]
                    courses_df.to_csv('data_files/courses.csv', index=False)
                    st.success(f"✅ Deleted {len(delete_data)} course(s)")
                    st.session_state.delete_mode = False
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel", key="cancel_delete_course"):
                    st.session_state.delete_mode = False
                    st.rerun()
        
        with col3:
            if st.button("📥 Download CSV", key="download_courses"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Courses",
                    data=csv,
                    file_name=f"courses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    elif data_type == "Lecturers":
        df = pd.read_csv('data_files/lecturers.csv')
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        st.subheader(f"Lecturers ({len(df)} entries)")
        
        # Row selection
        event = st.dataframe(df, use_container_width=True, height=400, on_select="rerun", key="lecturers_df")
        selected_rows = event.selection.rows
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Edit Selected", key="edit_lecturer"):
                if selected_rows:
                    st.session_state.edit_mode = True
                    st.session_state.edit_data_type = 'lecturers'
                    st.session_state.edit_row_index = selected_rows[0]
                    st.session_state.edit_data = df.iloc[selected_rows[0]].to_dict()
                    st.rerun()
                else:
                    st.info("Select a row to edit")
        
        with col2:
            if st.button("🗑️ Delete Selected", key="delete_lecturer"):
                if selected_rows:
                    st.session_state.delete_mode = True
                    st.session_state.delete_data_type = 'lecturers'
                    st.session_state.delete_rows = selected_rows
                    st.session_state.delete_data = df.iloc[selected_rows]
                    st.rerun()
                else:
                    st.info("Select a row to delete")
        
        # Edit mode
        if st.session_state.get('edit_mode', False) and st.session_state.get('edit_data_type') == 'lecturers':
            st.subheader("Edit Lecturer")
            edit_data = st.session_state.edit_data
            
            with st.form("edit_lecturer_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Lecturer Name:", value=edit_data['LecturerName'], key="edit_lecturer_name")
                    rank = st.selectbox("Rank", ["Professor", "Associate Professor", "Senior Lecturer", "Lecturer"], 
                                       index=0 if edit_data['Rank'] == 'Professor' else 
                                             1 if edit_data['Rank'] == 'Associate Professor' else
                                             2 if edit_data['Rank'] == 'Senior Lecturer' else 3,
                                       key="edit_rank")
                with col2:
                    department = st.text_input("Department:", value=edit_data['Department'], key="edit_lecturer_department")
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    submitted = st.form_submit_button("💾 Save Changes", type="primary")
                with col_cancel:
                    cancelled = st.form_submit_button("❌ Cancel")
                
                if submitted:
                    # Update lecturer
                    lecturers_df = pd.read_csv('data_files/lecturers.csv')
                    lecturer_id = edit_data['LecturerID']
                    
                    # Update row
                    lecturers_df.loc[lecturers_df['LecturerID'] == lecturer_id, 'LecturerName'] = name
                    lecturers_df.loc[lecturers_df['LecturerID'] == lecturer_id, 'Rank'] = rank
                    lecturers_df.loc[lecturers_df['LecturerID'] == lecturer_id, 'Department'] = department
                    
                    lecturers_df.to_csv('data_files/lecturers.csv', index=False)
                    st.success("✅ Lecturer updated successfully!")
                    st.session_state.edit_mode = False
                    st.rerun()
                
                if cancelled:
                    st.session_state.edit_mode = False
                    st.rerun()
        
        # Delete confirmation
        if st.session_state.get('delete_mode', False) and st.session_state.get('delete_data_type') == 'lecturers':
            delete_data = st.session_state.delete_data
            st.warning(f"⚠️ Are you sure you want to delete {len(delete_data)} lecturer(s)?")
            
            # Show lecturers to be deleted
            st.dataframe(delete_data[['LecturerName', 'Rank', 'Department']], 
                        use_container_width=True, height=200)
            
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Yes, Delete", key="confirm_delete_lecturer", type="primary"):
                    # Perform deletion
                    lecturers_df = pd.read_csv('data_files/lecturers.csv')
                    lecturer_ids_to_delete = delete_data['LecturerID'].tolist()
                    lecturers_df = lecturers_df[~lecturers_df['LecturerID'].isin(lecturer_ids_to_delete)]
                    lecturers_df.to_csv('data_files/lecturers.csv', index=False)
                    st.success(f"✅ Deleted {len(delete_data)} lecturer(s)")
                    st.session_state.delete_mode = False
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel", key="cancel_delete_lecturer"):
                    st.session_state.delete_mode = False
                    st.rerun()
        
        with col3:
            if st.button("📥 Download CSV", key="download_lecturers"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Lecturers",
                    data=csv,
                    file_name=f"lecturers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    elif data_type == "Venues":
        df = pd.read_csv('data_files/venues.csv')
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        st.subheader(f"Venues ({len(df)} entries)")
        
        # Row selection
        event = st.dataframe(df, use_container_width=True, height=400, on_select="rerun", key="venues_df")
        selected_rows = event.selection.rows
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Edit Selected", key="edit_venue"):
                if selected_rows:
                    st.session_state.edit_mode = True
                    st.session_state.edit_data_type = 'venues'
                    st.session_state.edit_row_index = selected_rows[0]
                    st.session_state.edit_data = df.iloc[selected_rows[0]].to_dict()
                    st.rerun()
                else:
                    st.info("Select a row to edit")
        
        with col2:
            if st.button("🗑️ Delete Selected", key="delete_venue"):
                if selected_rows:
                    st.session_state.delete_mode = True
                    st.session_state.delete_data_type = 'venues'
                    st.session_state.delete_rows = selected_rows
                    st.session_state.delete_data = df.iloc[selected_rows]
                    st.rerun()
                else:
                    st.info("Select a row to delete")
        
        # Edit mode
        if st.session_state.get('edit_mode', False) and st.session_state.get('edit_data_type') == 'venues':
            st.subheader("Edit Venue")
            edit_data = st.session_state.edit_data
            
            with st.form("edit_venue_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Venue Name:", value=edit_data['VenueName'], key="edit_venue_name")
                    capacity = st.number_input("Capacity:", min_value=1, value=int(edit_data['Capacity']), key="edit_venue_capacity")
                with col2:
                    building = st.text_input("Building:", value=str(edit_data.get('Building', '')), key="edit_venue_building")
                    room_type = st.selectbox("Room Type", ["Lecture Hall", "Lab", "Seminar Room"], 
                                           index=0 if edit_data['RoomType'] == 'Lecture Hall' else 
                                                 1 if edit_data['RoomType'] == 'Lab' else 2,
                                           key="edit_venue_type")
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    submitted = st.form_submit_button("💾 Save Changes", type="primary")
                with col_cancel:
                    cancelled = st.form_submit_button("❌ Cancel")
                
                if submitted:
                    # Update venue
                    venues_df = pd.read_csv('data_files/venues.csv')
                    venue_id = edit_data['VenueID']
                    
                    # Update row
                    venues_df.loc[venues_df['VenueID'] == venue_id, 'VenueName'] = name
                    venues_df.loc[venues_df['VenueID'] == venue_id, 'Capacity'] = capacity
                    venues_df.loc[venues_df['VenueID'] == venue_id, 'Building'] = building
                    venues_df.loc[venues_df['VenueID'] == venue_id, 'RoomType'] = room_type
                    
                    venues_df.to_csv('data_files/venues.csv', index=False)
                    st.success("✅ Venue updated successfully!")
                    st.session_state.edit_mode = False
                    st.rerun()
                
                if cancelled:
                    st.session_state.edit_mode = False
                    st.rerun()
        
        # Delete confirmation
        if st.session_state.get('delete_mode', False) and st.session_state.get('delete_data_type') == 'venues':
            delete_data = st.session_state.delete_data
            st.warning(f"⚠️ Are you sure you want to delete {len(delete_data)} venue(s)?")
            
            # Show venues to be deleted
            st.dataframe(delete_data[['VenueName', 'Capacity', 'Building', 'RoomType']], 
                        use_container_width=True, height=200)
            
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Yes, Delete", key="confirm_delete_venue", type="primary"):
                    # Perform deletion
                    venues_df = pd.read_csv('data_files/venues.csv')
                    venue_ids_to_delete = delete_data['VenueID'].tolist()
                    venues_df = venues_df[~venues_df['VenueID'].isin(venue_ids_to_delete)]
                    venues_df.to_csv('data_files/venues.csv', index=False)
                    st.success(f"✅ Deleted {len(delete_data)} venue(s)")
                    st.session_state.delete_mode = False
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel", key="cancel_delete_venue"):
                    st.session_state.delete_mode = False
                    st.rerun()
        
        with col3:
            if st.button("📥 Download CSV", key="download_venues"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Venues",
                    data=csv,
                    file_name=f"venues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    elif data_type == "Time Slots":
        df = pd.read_csv('data_files/timeslots.csv')
        
        # Apply search filter
        if search_term:
            df = df[df.apply(lambda row: any(str(search_term).lower() in str(val).lower() for val in row), axis=1)]
        
        # Add action column
        df['Action'] = '🕐 Manage'
        
        st.subheader(f"Time Slots ({len(df)} entries)")
        
        # Row selection
        event = st.dataframe(df, use_container_width=True, height=400, on_select="rerun", key="timeslots_df")
        selected_rows = event.selection.rows
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Edit Selected", key="edit_slots"):
                if selected_rows:
                    st.info(f"Selected {len(selected_rows)} row(s) to edit")
                else:
                    st.info("Select a row to edit")
        
        with col2:
            if st.button("🗑️ Delete Selected", key="delete_slots"):
                if selected_rows:
                    st.info(f"Selected {len(selected_rows)} row(s) to delete")
                else:
                    st.info("Select a row to delete")
        
        with col3:
            if st.button("📥 Download CSV", key="download_slots"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Time Slots",
                    data=csv,
                    file_name=f"timeslots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SMART Timetable Scheduler v1.0 | ML-Guided Intelligent Scheduling</p>
</div>
""", unsafe_allow_html=True)
