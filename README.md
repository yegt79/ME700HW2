# Assignment 2 - Numerical Methods Repository

This repository contains several modules implementing different numerical methods, with a focus on the **Direct Stiffness Method**. It is organized into the following folders:

- *files*: **Main** source code for the project.
- *Direct_Stiffness_Method*: Implementation of the **Direct Stiffness Method**.
- *functions*: **Mathematical functions** required to support the code.
- *examples*: Example problems with solutions, where you can insert your own inputs or explore previous problems.
- *test_Direct_Stiffness_Method*: Test suite using **pytest**, covering more than $90\%$ of the codebase.

## Downloading the Code from GitHub

To get started, you'll need to download the code from this GitHub repository. Follow these steps to clone it to your local machine:

1. **Install Git**: If you don’t have Git installed, download and install it from *git-scm.com*.
2. **Clone the Repository**: Open a terminal or command prompt and run:
   git clone https://github.com/your-username/your-repository-name.git
   Replace *your-username* with your GitHub username and *your-repository-name* with the name of this repository.
3. **Navigate to the Repository**: Move into the project folder:
   cd your-repository-name

Now you're ready to set up the environment and start using the code!

## Setup Instructions

To run the code, follow the steps below.

### 1. Set up the Environment

Ensure you have **Anaconda** installed on your machine. If not, download it from *anaconda.com*.

#### Creating and Activating the Virtual Environment

1. Open a *Terminal* session.
2. Create a virtual environment with **Conda**:
   conda create --name setup-examples-env python=3.9.13
3. Activate the virtual environment:
   conda activate setup-examples-env
4. Verify the Python version (should be *3.9.13*):
   python --version
5. Update base modules:
   pip install --upgrade pip setuptools wheel

*Note*: After the environment is created, you can activate it anytime with:
conda activate setup-examples-env

#### Installing Dependencies

1. Navigate to the project folder:
   cd your-repository-name
2. Ensure the *pyproject.toml* file is present:
   dir  # On Windows
   ls   # On Unix-based systems (use *dir* instead if on Windows)
3. Install the project in editable mode:
   pip install -e .

### 2. Running the Code

Once the environment is set up and dependencies are installed, you can run the code from the *examples* folder or integrate your own inputs into the provided scripts.

### 3. Setting up Code Coverage (Optional)

If you'd like to check code coverage, use the provided *.yml* configuration file. This file is compatible with code coverage tools (e.g., **pytest-cov**). To run it:
1. Install the coverage tool:
   pip install pytest-cov
2. Run the tests with coverage:
   pytest --cov=./ --cov-report=html
3. Open the generated *htmlcov/index.html* file in a browser to view the coverage report.

## Folder Structure

- *files*: **Core** source code.
- *Direct_Stiffness_Method*: **Direct Stiffness Method** implementation.
- *functions*: Supporting **mathematical functions**.
- *examples*: **Tutorials** and example problems.
- *test_Direct_Stiffness_Method*: **Automated tests** with >$90\%$ coverage.

## Requirements

- **Anaconda** (for environment management)
- *Python 3.9.13*
- **Git** (for cloning the repository)
- *pytest* (for running tests)
- *pytest-cov* (optional, for code coverage)
- *SciPy* (for numerical computations)

