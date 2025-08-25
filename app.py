import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import json
from typing import Dict, Any
import traceback

from network_optimizer import NetworkOptimizer
from utils import validate_csv_structure, get_sample_data_info

# Configure page
st.set_page_config(
    page_title="Provider Network Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏥 Provider Network Optimizer Dashboard")
    st.markdown("Upload provider and member CSV files to optimize your healthcare network")
    
    # Initialize session state
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # Display sample data requirements
        with st.expander("📋 Required CSV Format", expanded=False):
            st.markdown(get_sample_data_info())
        
        # File upload sections
        st.subheader("Provider Data")
        provider_file = st.file_uploader(
            "Upload Provider CSV",
            type=['csv'],
            key="provider_upload",
            help="CSV file containing provider information"
        )
        
        st.subheader("Member Data")
        member_file = st.file_uploader(
            "Upload Member CSV",
            type=['csv'],
            key="member_upload",
            help="CSV file containing member information"
        )
        
        # Configuration options
        st.header("⚙️ Configuration")
        
        max_drive_time = st.slider(
            "Max Drive Time (minutes)",
            min_value=10,
            max_value=60,
            value=30,
            step=5,
            help="Maximum acceptable drive time from member to provider"
        )
        
        min_coverage = st.slider(
            "Minimum Coverage (%)",
            min_value=80.0,
            max_value=100.0,
            value=95.0,
            step=1.0,
            help="Minimum percentage of members that must be covered"
        )
        
        min_rating = st.slider(
            "Minimum Average Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Minimum acceptable average provider rating"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if provider_file is not None and member_file is not None:
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                run_optimization(provider_file, member_file, max_drive_time, min_coverage, min_rating)
        else:
            st.info("👆 Please upload both provider and member CSV files to begin optimization")
    
    with col2:
        if st.session_state.optimization_results is not None:
            if st.button("📥 Export Results", use_container_width=True):
                export_results()
    
    # Display results if available
    if st.session_state.optimization_results is not None:
        display_results()
    
    # Display file previews
    display_file_previews(provider_file, member_file)

def run_optimization(provider_file, member_file, max_drive_time, min_coverage, min_rating):
    """Run the network optimization process"""
    with st.spinner("🔄 Processing data and running optimization..."):
        try:
            # Read and validate CSV files
            providers_df = pd.read_csv(provider_file)
            members_df = pd.read_csv(member_file)
            
            # Validate CSV structure
            provider_validation = validate_csv_structure(providers_df, 'provider')
            member_validation = validate_csv_structure(members_df, 'member')
            
            if not provider_validation['valid']:
                st.error(f"❌ Provider CSV validation failed: {provider_validation['message']}")
                return
            
            if not member_validation['valid']:
                st.error(f"❌ Member CSV validation failed: {member_validation['message']}")
                return
            
            # Initialize optimizer with configuration
            optimizer = NetworkOptimizer(
                max_drive_min=max_drive_time,
                min_coverage_pct=min_coverage,
                min_avg_rating=min_rating
            )
            
            # Run optimization with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_container = st.container()
            
            status_text.text("🔍 Analyzing provider data...")
            progress_bar.progress(20)
            
            status_text.text("👥 Processing member data...")
            progress_bar.progress(40)
            
            status_text.text("🧮 Building candidate pairs...")
            progress_bar.progress(60)
            
            status_text.text("⚡ Running optimization algorithm...")
            progress_bar.progress(80)
            
            # Add a live log container for real-time updates
            log_container = progress_container.empty()
            
            class StreamlitProgressHandler:
                def __init__(self, log_container, status_text):
                    self.log_container = log_container
                    self.status_text = status_text
                    self.removed_count = 0
                    self.logs = []
                
                def update_progress(self, message):
                    if "Removed provider" in message:
                        self.removed_count += 1
                        self.status_text.text(f"🔧 Optimizing network... Removed {self.removed_count} providers")
                        # Keep only last 10 log entries
                        self.logs.append(message)
                        if len(self.logs) > 10:
                            self.logs.pop(0)
                        self.log_container.text("\n".join(self.logs[-5:]))  # Show last 5 entries
            
            # Create progress handler and pass to optimizer
            progress_handler = StreamlitProgressHandler(log_container, status_text)
            optimizer.progress_handler = progress_handler
            
            results = optimizer.optimize(members_df, providers_df)
            
            status_text.text("✅ Optimization complete!")
            progress_bar.progress(100)
            
            # Store results in session state
            st.session_state.optimization_results = results
            st.session_state.processed_data = {
                'providers': providers_df,
                'members': members_df,
                'config': {
                    'max_drive_time': max_drive_time,
                    'min_coverage': min_coverage,
                    'min_rating': min_rating
                }
            }
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 Network optimization completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during optimization: {str(e)}")
            st.expander("🔍 Error Details", expanded=False).code(traceback.format_exc())

def display_results():
    """Display optimization results with visualizations"""
    results = st.session_state.optimization_results
    data = st.session_state.processed_data
    
    st.header("📊 Optimization Results")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Coverage",
            f"{results['coverage_pct']:.1f}%",
            help="Percentage of members assigned to providers"
        )
    
    with col2:
        st.metric(
            "Average Rating",
            f"{results['avg_rating']:.2f}",
            help="Average CMS rating of selected providers"
        )
    
    with col3:
        st.metric(
            "Total Cost",
            f"${results['total_cost']:,.0f}",
            help="Total annual cost of selected providers"
        )
    
    with col4:
        st.metric(
            "Providers Used",
            f"{results['providers_used']}",
            help="Number of providers in optimized network"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "🗺️ Geographic View", "📋 Assignments", "📊 Provider Details"])
    
    with tab1:
        display_analytics(results, data)
    
    with tab2:
        display_geographic_view(results, data)
    
    with tab3:
        display_assignments(results)
    
    with tab4:
        display_provider_details(results, data)

def display_analytics(results, data):
    """Display analytics charts and insights"""
    assignments = results['final_assignments']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Provider type distribution
        if not assignments.empty:
            type_counts = assignments['ProviderType'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Provider Distribution by Type"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Rating distribution
        if not assignments.empty:
            fig_hist = px.histogram(
                assignments,
                x='CMS_Rating',
                nbins=10,
                title="Provider Rating Distribution",
                labels={'CMS_Rating': 'CMS Rating', 'count': 'Number of Providers'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Cost analysis
    if not assignments.empty:
        provider_costs = assignments.groupby('ProviderId').agg({
            'Cost': 'first',
            'MemberId': 'count',
            'CMS_Rating': 'first'
        }).rename(columns={'MemberId': 'Members_Assigned'})
        
        provider_costs['Cost_Per_Member'] = provider_costs['Cost'] / provider_costs['Members_Assigned']
        
        fig_scatter = px.scatter(
            provider_costs,
            x='Members_Assigned',
            y='Cost_Per_Member',
            color='CMS_Rating',
            size='Cost',
            title="Provider Cost Efficiency Analysis",
            labels={
                'Members_Assigned': 'Members Assigned',
                'Cost_Per_Member': 'Cost per Member ($)',
                'CMS_Rating': 'CMS Rating'
            },
            hover_data={'Cost': ':,.0f'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

def display_geographic_view(results, data):
    """Display geographic visualization of assignments"""
    assignments = results['final_assignments']
    providers = data['providers']
    members = data['members']
    
    if assignments.empty:
        st.warning("No assignments to display on map")
        return
    
    # Get assigned providers with coordinates
    assigned_providers = providers[providers['ProviderId'].isin(assignments['ProviderId'].unique())].copy()
    
    if 'Latitude' not in assigned_providers.columns or 'Longitude' not in assigned_providers.columns:
        st.warning("Provider coordinates not available for geographic visualization")
        return
    
    # Check for color column (ProviderType vs Type)
    color_column = None
    if 'ProviderType' in assigned_providers.columns:
        color_column = 'ProviderType'
    elif 'Type' in assigned_providers.columns:
        color_column = 'Type'
    
    # Create map - check for correct column names
    hover_columns = []
    if 'ProviderId' in assigned_providers.columns:
        hover_columns.append('ProviderId')
    if 'CMS_Rating' in assigned_providers.columns:
        hover_columns.append('CMS_Rating')
    elif 'CMS Rating' in assigned_providers.columns:
        hover_columns.append('CMS Rating')
    
    # Create the map
    if color_column:
        fig_map = px.scatter_mapbox(
            assigned_providers,
            lat='Latitude',
            lon='Longitude',
            color=color_column,
            size='Cost',
            hover_data=hover_columns,
            title="Geographic Distribution of Selected Providers",
            mapbox_style="open-street-map",
            height=600
        )
    else:
        # Fallback without color if no type column available
        fig_map = px.scatter_mapbox(
            assigned_providers,
            lat='Latitude',
            lon='Longitude',
            size='Cost',
            hover_data=hover_columns,
            title="Geographic Distribution of Selected Providers",
            mapbox_style="open-street-map",
            height=600
        )
    
    fig_map.update_layout(
        mapbox_center_lat=assigned_providers['Latitude'].mean(),
        mapbox_center_lon=assigned_providers['Longitude'].mean(),
        mapbox_zoom=8
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

def display_assignments(results):
    """Display member-provider assignments table"""
    assignments = results['final_assignments']
    
    if assignments.empty:
        st.warning("No assignments to display")
        return
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_rating = assignments['CMS_Rating'].mean()
        st.metric("Avg Provider Rating", f"{avg_rating:.2f}")
    
    with col2:
        total_members = len(assignments)
        st.metric("Total Assignments", f"{total_members:,}")
    
    with col3:
        unique_providers = assignments['ProviderId'].nunique()
        st.metric("Unique Providers", f"{unique_providers}")
    
    # Assignments table with filtering
    st.subheader("Assignment Details")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        provider_types = ['All'] + sorted(assignments['ProviderType'].unique().tolist())
        selected_type = st.selectbox("Filter by Provider Type", provider_types)
    
    with col2:
        min_rating_filter = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.1)
    
    with col3:
        max_cost_filter = st.number_input(
            "Maximum Cost", 
            min_value=0,
            max_value=int(assignments['Cost'].max()),
            value=int(assignments['Cost'].max())
        )
    
    # Apply filters
    filtered_assignments = assignments.copy()
    
    if selected_type != 'All':
        filtered_assignments = filtered_assignments[filtered_assignments['ProviderType'] == selected_type]
    
    filtered_assignments = filtered_assignments[
        (filtered_assignments['CMS_Rating'] >= min_rating_filter) &
        (filtered_assignments['Cost'] <= max_cost_filter)
    ]
    
    # Display filtered table
    st.dataframe(
        filtered_assignments[['MemberId', 'ProviderId', 'ProviderType', 'CMS_Rating', 'Cost']],
        use_container_width=True,
        hide_index=True
    )
    
    # Download filtered data
    if not filtered_assignments.empty:
        csv_data = filtered_assignments.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Assignments",
            data=csv_data,
            file_name="filtered_assignments.csv",
            mime="text/csv"
        )

def display_provider_details(results, data):
    """Display detailed provider information"""
    assignments = results['final_assignments']
    
    if assignments.empty:
        st.warning("No provider details to display")
        return
    
    # Provider utilization analysis
    provider_stats = assignments.groupby(['ProviderId', 'ProviderType', 'CMS_Rating', 'Cost']).size().reset_index(name='Members_Assigned')
    
    # Sort by members assigned
    provider_stats = provider_stats.sort_values('Members_Assigned', ascending=False)
    
    st.subheader("Provider Utilization")
    
    # Top providers chart
    top_providers = provider_stats.head(15)
    fig_bar = px.bar(
        top_providers,
        x='ProviderId',
        y='Members_Assigned',
        color='CMS_Rating',
        title="Top 15 Providers by Member Assignment",
        labels={'ProviderId': 'Provider ID', 'Members_Assigned': 'Members Assigned'}
    )
    fig_bar.update_layout(xaxis={'tickangle': 45})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed provider table
    st.subheader("Provider Summary")
    provider_stats['Cost_Per_Member'] = provider_stats['Cost'] / provider_stats['Members_Assigned']
    
    # Format columns for display
    display_stats = provider_stats.copy()
    display_stats['Cost'] = display_stats['Cost'].apply(lambda x: f"${x:,.0f}")
    display_stats['Cost_Per_Member'] = display_stats['Cost_Per_Member'].apply(lambda x: f"${x:,.0f}")
    display_stats['CMS_Rating'] = display_stats['CMS_Rating'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_stats[['ProviderId', 'ProviderType', 'CMS_Rating', 'Members_Assigned', 'Cost', 'Cost_Per_Member']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'ProviderId': 'Provider ID',
            'ProviderType': 'Type',
            'CMS_Rating': 'Rating',
            'Members_Assigned': 'Members',
            'Cost': 'Annual Cost',
            'Cost_Per_Member': 'Cost/Member'
        }
    )

def display_file_previews(provider_file, member_file):
    """Display previews of uploaded files"""
    if provider_file is not None or member_file is not None:
        st.header("📋 Data Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if provider_file is not None:
                st.subheader("Provider Data Preview")
                try:
                    provider_df = pd.read_csv(provider_file)
                    st.dataframe(provider_df.head(10), use_container_width=True)
                    st.caption(f"Shape: {provider_df.shape[0]} rows × {provider_df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error reading provider file: {str(e)}")
        
        with col2:
            if member_file is not None:
                st.subheader("Member Data Preview")
                try:
                    member_df = pd.read_csv(member_file)
                    st.dataframe(member_df.head(10), use_container_width=True)
                    st.caption(f"Shape: {member_df.shape[0]} rows × {member_df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error reading member file: {str(e)}")

def export_results():
    """Export optimization results to downloadable formats"""
    if st.session_state.optimization_results is None:
        st.error("No results to export")
        return
    
    results = st.session_state.optimization_results
    
    # Create export data
    assignments = results['final_assignments']
    
    if assignments.empty:
        st.warning("No assignments to export")
        return
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = assignments.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name="network_optimization_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON export with summary
        export_data = {
            'optimization_summary': {
                'coverage_pct': results['coverage_pct'],
                'avg_rating': results['avg_rating'],
                'total_cost': results['total_cost'],
                'providers_used': results['providers_used']
            },
            'assignments': assignments.to_dict('records')
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="📊 Download as JSON",
            data=json_data,
            file_name="network_optimization_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.success("✅ Export options ready!")

if __name__ == "__main__":
    main()
