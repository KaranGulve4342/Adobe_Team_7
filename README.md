# Curvetopia

## Overview

Curvetopia is the Adobe GenSolve project for our group, aimed at generating a model that can identify simple shape doodles. This project includes a well-structured setup process, ensuring ease of use and reproducibility. Users can clone the repository, set up a virtual environment, and manage datasets efficiently.

Key features of Curvetopia include:

- **Shape Identification:** Generate and train a model to identify and distinguish between simple shape doodles.
- **Data Management:** Easily manage and organize datasets using the designated `datasets` folder.
- **Virtual Environment:** Isolate project dependencies using a virtual environment.
- **Dependency Management:** Install necessary dependencies with a single command.
- **Cloud Integration:** Store and retrieve machine learning models using cloud storage solutions.

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/2manas8/Curvetopia.git
cd Curvetopia
```

### 2. Create the 'datasets' Folder

Create a folder named **`datasets`** in the root directory of the project. This is where you will store the extracted ZIP files containing the datasets.

```bash
mkdir datasets
```

### 3. Create and Activate a Virtual Environment

Create a virtual environment to isolate the project's dependencies. Run the following command to create and activate the virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For Unix/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Extract the Datasets

Extract the dataset ZIP files into the **`datasets`** folder. Ensure that the structure is maintained as expected by the project.

```bash
unzip path/to/your_dataset.zip -d datasets/
```

### 5. Install Dependencies

Install the required dependencies for the project. This may involve using a package manager like **`pip`** for Python projects:

```bash
pip install -r requirements.txt
```