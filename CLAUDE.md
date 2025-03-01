# CLAUDE.md - IJCNN-2025 Project Guidelines

## Build/Lint/Test Commands
- Run Python scripts: `python scripts/hypersphere.py`
- Run visualization analysis: `python scripts/mnist_alphabetic_character_analysis.py` (outputs saved to `figures/` directory)
- No specific lint/test commands identified in codebase

## Code Style Guidelines
- **Imports**: Group standard library imports first, followed by third-party libraries 
- **Formatting**: Use descriptive variable names and function names with snake_case
- **Types**: Use numpy arrays for data processing with appropriate data types
- **Documentation**: Include docstrings for functions with parameters and return descriptions
- **Error Handling**: Use try/except blocks with specific exception types
- **Plotting**: Use matplotlib for visualizations with consistent styling
- **Data Structure**: Work with numpy arrays where:
  - Columns 0-9: Softmax outputs
  - Column 10: True class
  - Column 11: Predicted class
  - Columns 12-13: Distances to centroids