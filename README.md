$Assignment\ 2\ -\ Numerical\ Methods\ Repository$

This repository contains several modules implementing different numerical methods, with a focus on the $\textbf{Direct Stiffness Method}$. It is organized into the following folders:

- $\textit{files}$: $\textbf{Main}$ source code for the project.
- $\textit{Direct_Stiffness_Method}$: Implementation of the $\textbf{Direct Stiffness Method}$.
- $\textit{functions}$: $\textbf{Mathematical functions}$ required to support the code.
- $\textit{examples}$: Example problems with solutions, where you can insert your own inputs or explore previous problems.
- $\textit{test_Direct_Stiffness_Method}$: Test suite using $\textbf{pytest}$, covering more than $90\%$ of the codebase.

$\textbf{Downloading the Code from GitHub}$

To get started, you'll need to download the code from this GitHub repository. Follow these steps to clone it to your local machine:

1. $\textbf{Install Git}$: If you don’t have Git installed, download and install it from $\textit{git-scm.com}$.
2. $\textbf{Clone the Repository}$: Open a terminal or command prompt and run:
   git clone https://github.com/your-username/your-repository-name.git
   Replace $\textit{your-username}$ with your GitHub username and $\textit{your-repository-name}$ with the name of this repository.
3. $\textbf{Navigate to the Repository}$: Move into the project folder:
   cd your-repository-name

Now you're ready to set up the environment and start using the code!

$\textbf{Setup Instructions}$

To run the code, follow the steps below.

1. $\textbf{Set up the Environment}$

Ensure you have $\textbf{Anaconda}$ installed on your machine. If not, download it from $\textit{anaconda.com}$.

$\textbf{Creating and Activating the Virtual Environment}$

1. Open a $\textit{Terminal}$ session.
2. Create a virtual environment with $\textbf{Conda}$:
   conda create --name setup-examples-env python=3.9.13
3. Activate the virtual environment:
   conda activate setup-examples-env
4. Verify the Python version (should be $\textit{3.9.13}$):
   python --version
5. Update base modules:
   pip install --upgrade pip setuptools wheel

$\textit{Note}$: After the environment is created, you can activate it anytime with:
conda activate setup-examples-env

$\textbf{Installing Dependencies}$

1. Navigate to the project folder:
   cd your-repository-name
2. Ensure the $\textit{pyproject.toml}$ file is present:
   dir  # On Windows
   ls   # On Unix-based systems (use $\textit{dir}$ instead if on Windows)
3. Install the project in editable mode:
   pip install -e .

2. $\textbf{Running the Code}$

Once the environment is set up and dependencies are installed, you can run the code from the $\textit{examples}$ folder or integrate your own inputs into the provided scripts.

3. $\textbf{Setting up Code Coverage (Optional)}$

If you'd like to check code coverage, use the provided $\textit{.yml}$ configuration file. This file is compatible with code coverage tools (e.g., $\textbf{pytest-cov}$). To run it:
1. Install the coverage tool:
   pip install pytest-cov
2. Run the tests with coverage:
   pytest --cov=./ --cov-report=html
3. Open the generated $\textit{htmlcov/index.html}$ file in a browser to view the coverage report.

$\textbf{Folder Structure}$

- $\textit{files}$: $\textbf{Core}$ source code.
- $\textit{Direct_Stiffness_Method}$: $\textbf{Direct Stiffness Method}$ implementation.
- $\textit{functions}$: Supporting $\textbf{mathematical functions}$.
- $\textit{examples}$: $\textbf{Tutorials}$ and example problems.
- $\textit{test_Direct_Stiffness_Method}$: $\textbf{Automated tests}$ with $>90\%$ coverage.

$\textbf{Requirements}$

- $\textbf{Anaconda}$ (for environment management)
- $\textit{Python 3.9.13}$
- $\textbf{Git}$ (for cloning the repository)
- $\textit{pytest}$ (for running tests)
- $\textit{pytest-cov}$ (optional, for code coverage)
- $\textit{SciPy}$ (for numerical computations)

$\textbf{Notes}$

- Replace $\textit{your-username}$ and $\textit{your-repository-name}$ with the actual GitHub username and repository name when you publish this.
- I’ve added $\textit{SciPy}$ to the requirements list, assuming it’s used for numerical computations in your project (common for methods like the $\textbf{Direct Stiffness Method}$). If you need a specific version of $\textit{SciPy}$, let me know, and I can update it!
