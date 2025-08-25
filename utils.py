import pandas as pd
from typing import Dict, Any, List

def validate_csv_structure(df: pd.DataFrame, file_type: str) -> Dict[str, Any]:
    """
    Validate CSV file structure for provider or member data
    
    Args:
        df: DataFrame to validate
        file_type: 'provider' or 'member'
    
    Returns:
        Dict with 'valid' boolean and 'message' string
    """
    if df.empty:
        return {'valid': False, 'message': f'{file_type} file is empty'}
    
    if file_type == 'provider':
        return _validate_provider_structure(df)
    elif file_type == 'member':
        return _validate_member_structure(df)
    else:
        return {'valid': False, 'message': f'Unknown file type: {file_type}'}

def _validate_provider_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate provider CSV structure"""
    required_columns = {
        'id': ['providerid', 'provider_id', 'provider id', 'id'],
        'latitude': ['latitude', 'lat', 'latt'],
        'longitude': ['longitude', 'lon', 'lng']
    }
    
    optional_columns = {
        'rating': ['cmsrating', 'cms_rating', 'cms rating', 'rating', 'star'],
        'cost': ['cost', 'annualcost', 'annual_cost', 'contractcost', 'contract_cost'],
        'availability': ['availability', 'capacity', 'avail'],
        'type': ['type', 'facilitytype', 'facility_type'],
        'source': ['source']
    }
    
    # Check for required columns
    missing_required = []
    df_columns_lower = [col.lower().replace(' ', '').replace('_', '') for col in df.columns]
    
    for field, candidates in required_columns.items():
        found = False
        for candidate in candidates:
            if candidate.lower().replace(' ', '').replace('_', '') in df_columns_lower:
                found = True
                break
        if not found:
            missing_required.append(field)
    
    if missing_required:
        return {
            'valid': False, 
            'message': f'Missing required provider columns: {missing_required}. Available columns: {list(df.columns)}'
        }
    
    # Validate data types for coordinates
    lat_col = _find_column(df, required_columns['latitude'])
    lon_col = _find_column(df, required_columns['longitude'])
    
    try:
        pd.to_numeric(df[lat_col], errors='raise')
        pd.to_numeric(df[lon_col], errors='raise')
    except (ValueError, TypeError):
        return {
            'valid': False,
            'message': 'Latitude and longitude columns must contain numeric values'
        }
    
    return {'valid': True, 'message': 'Provider CSV structure is valid'}

def _validate_member_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate member CSV structure"""
    required_columns = {
        'id': ['memberid', 'member_id', 'member id', 'id'],
        'latitude': ['latitude', 'lat'],
        'longitude': ['longitude', 'lon', 'lng']
    }
    
    # Check for required columns
    missing_required = []
    df_columns_lower = [col.lower().replace(' ', '').replace('_', '') for col in df.columns]
    
    for field, candidates in required_columns.items():
        found = False
        for candidate in candidates:
            if candidate.lower().replace(' ', '').replace('_', '') in df_columns_lower:
                found = True
                break
        if not found:
            missing_required.append(field)
    
    if missing_required:
        return {
            'valid': False,
            'message': f'Missing required member columns: {missing_required}. Available columns: {list(df.columns)}'
        }
    
    # Validate data types for coordinates
    lat_col = _find_column(df, required_columns['latitude'])
    lon_col = _find_column(df, required_columns['longitude'])
    
    try:
        pd.to_numeric(df[lat_col], errors='raise')
        pd.to_numeric(df[lon_col], errors='raise')
    except (ValueError, TypeError):
        return {
            'valid': False,
            'message': 'Latitude and longitude columns must contain numeric values'
        }
    
    return {'valid': True, 'message': 'Member CSV structure is valid'}

def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Find column name from candidates"""
    df_columns_lower = {col.lower().replace(' ', '').replace('_', ''): col for col in df.columns}
    
    for candidate in candidates:
        key = candidate.lower().replace(' ', '').replace('_', '')
        if key in df_columns_lower:
            return df_columns_lower[key]
    
    # Fallback to exact match
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # Return first column name as fallback to ensure we always return a string
    return df.columns[0] if len(df.columns) > 0 else ""

def get_sample_data_info() -> str:
    """Get sample data format information"""
    return """
    ## Required CSV Format

    ### Provider Data
    **Required columns** (case-insensitive):
    - `ProviderId` or `provider_id` or `id`: Unique provider identifier
    - `Latitude` or `lat`: Provider latitude coordinate
    - `Longitude` or `lon` or `lng`: Provider longitude coordinate

    **Optional columns**:
    - `CMS_Rating` or `rating` or `star`: Provider quality rating (1-5)
    - `Cost` or `annual_cost`: Annual provider cost
    - `Availability` or `capacity`: Provider capacity/availability
    - `Type` or `facility_type`: Provider type/category
    - `Source`: Data source information

    ### Member Data  
    **Required columns** (case-insensitive):
    - `MemberId` or `member_id` or `id`: Unique member identifier
    - `Latitude` or `lat`: Member latitude coordinate  
    - `Longitude` or `lon` or `lng`: Member longitude coordinate

    ### Data Tips
    - Coordinates should be in decimal degrees format
    - Provider costs should be numeric values
    - Ratings should be between 1-5
    - Missing optional fields will use reasonable defaults
    """

def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"

def format_rating(value: float) -> str:
    """Format rating value"""
    return f"{value:.2f}"

def get_provider_type_colors() -> Dict[str, str]:
    """Get color mapping for provider types"""
    return {
        'hospital': '#FF6B6B',
        'clinic': '#4ECDC4', 
        'nursing_home': '#45B7D1',
        'scan_center': '#96CEB4',
        'supplier_directory': '#FFEAA7',
        'other': '#DDA0DD'
    }

def create_summary_stats(assignments: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for assignments"""
    if assignments.empty:
        return {
            'total_assignments': 0,
            'unique_members': 0,
            'unique_providers': 0,
            'avg_rating': 0.0,
            'total_cost': 0.0
        }
    
    return {
        'total_assignments': len(assignments),
        'unique_members': assignments['MemberId'].nunique(),
        'unique_providers': assignments['ProviderId'].nunique(),
        'avg_rating': assignments['CMS_Rating'].mean() if 'CMS_Rating' in assignments.columns else 0.0,
        'total_cost': assignments['Cost'].sum() if 'Cost' in assignments.columns else 0.0
    }
