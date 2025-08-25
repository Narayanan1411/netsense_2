# Provider Network Optimizer

## Overview

The Provider Network Optimizer is a healthcare analytics dashboard built with Streamlit that helps optimize provider networks by analyzing spatial relationships between healthcare providers and members. The application processes CSV files containing provider and member data to determine optimal network configurations while maintaining coverage requirements and quality standards. It uses advanced algorithms including k-d trees for spatial queries and intelligent reassignment strategies to minimize network costs while ensuring adequate member access to care.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based dashboard using Streamlit for the user interface
- **Interactive Visualizations**: Plotly integration for creating dynamic charts and maps
- **File Upload System**: CSV file processing with real-time validation and feedback
- **Session State Management**: Persistent storage of optimization results and processed data across user interactions

### Backend Architecture
- **NetworkOptimizer Core Engine**: Main optimization logic using spatial algorithms and constraint satisfaction
- **Data Processing Pipeline**: Pandas-based data manipulation with column mapping and validation
- **Numba JIT Compilation**: Performance optimization for computationally intensive operations
- **Spatial Computing**: cKDTree implementation for efficient nearest-neighbor searches

### Data Storage Solutions
- **In-Memory Processing**: Session-based data storage using Streamlit's session state
- **CSV File Processing**: Direct file upload and processing without persistent database storage
- **Temporary Data Structures**: NumPy arrays and Pandas DataFrames for runtime computations

### Authentication and Authorization
- **No Authentication System**: Direct access application without user management or access controls

### Algorithm Design Patterns
- **Incremental Reassignment**: Intelligent provider removal with member reassignment to maintain coverage
- **Constraint Optimization**: Multi-objective optimization balancing cost reduction, coverage requirements, and quality thresholds
- **Spatial Indexing**: K-d tree data structure for O(log n) provider lookup performance
- **Type-Aware Processing**: Provider categorization system with minimum count requirements per type

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing for array operations and mathematical calculations
- **SciPy**: Scientific computing library specifically using cKDTree for spatial operations

### Visualization Libraries
- **Plotly Express**: High-level plotting interface for charts and graphs
- **Plotly Graph Objects**: Low-level plotting interface for custom visualizations

### Performance Libraries
- **Numba**: Just-in-time compilation for performance-critical numerical functions

### Standard Libraries
- **JSON**: Data serialization for configuration and results storage
- **IO**: Input/output operations for file handling
- **Logging**: Application logging and debugging
- **Traceback**: Error handling and debugging support
- **Typing**: Type hints for better code documentation and IDE support

### Data Processing
- **CSV File Support**: Direct processing of comma-separated value files
- **Column Mapping System**: Flexible column name matching for various CSV formats
- **Data Validation**: Built-in validation for required fields and data types