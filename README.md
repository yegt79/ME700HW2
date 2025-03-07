# ME700HW2 - Direct Stiffness Method Implementation

This repository contains my implementation of the Direct Stiffness Method for ME700 Assignment 2, written in Python. It models beam structures in 3D, handles boundary conditions, and performs buckling analysis, with a comprehensive test suite achieving over 90% code coverage.

## Features

- **BeamComponent**: Represents beam elements with nodes, elements, and material properties (E, ν, A, Iy, Iz, J).
- **BoundaryCondition**: Manages fixed nodes and applied loads for static and buckling analysis.
- **BeamSolver**: Solves for displacements, reactions, and buckling modes using the direct stiffness method.
- **Unit Tests**: Extensive tests in `test_Direct_Stiffness_Method.py` cover input validation, matrix operations, and solver functionality.

## Installation

To run the code and tests, install the following dependencies. Instructions are provided for macOS/Linux (using a terminal) and Windows (using Command Prompt or PowerShell). A virtual environment is recommended.

### Dependencies
- **Python 3.13**: Core language (tested with 3.13.1 and 3.13.2).
- **NumPy**: For matrix operations and numerical computations.
- **pytest**: For running unit tests.
- **pytest-cov**: For test coverage reporting.

### Installation Steps

1. **Install Python 3.13**
   - **macOS/Linux**: 
     - Use Homebrew: `brew install python@3.13` or download from [python.org](https://www.python.org/downloads/).
     - Verify: `python3 --version` (should show 3.13.x).
   - **Windows**: 
     - Download from [python.org](https://www.python.org/downloads/), install with "Add Python to PATH" checked.
     - Verify: `python --version` or `py -3.13 --version`.

2. **Set Up a Virtual Environment** (Recommended)
   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
