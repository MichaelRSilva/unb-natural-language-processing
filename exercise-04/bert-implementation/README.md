# BERT Text Classification

## Project Overview
This project implements a BERT-based text classification pipeline using `transformers`, `torch`, and `sklearn`. The code is modular and organized into separate files for data loading, tokenization, dataset handling, training, and evaluation.

## Project Structure
```
📂 bert_project
 ├── 📄 load_data.py                    # Loads data from CSV with dynamic file and column names
 ├── 📄 tokenizer.py                    # Handles tokenization
 ├── 📄 bert_dataset.py                 # Defines the dataset class
 ├── 📄 train.py                        # Training functions
 ├── 📄 evaluate.py                     # Evaluation functions
 ├── 📄 bert_exercise_dmoz_health.ipynb # Jupyter Notebook to run the dmoz health dataset
 ├── 📄 README.md                       # Documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/MichaelRSilva/unb-natural-language-processing.git
   cd exercise-04
   ```
2. Install dependencies:
   ```sh
   pip install torch transformers scikit-learn pandas matplotlib
   ```

## Usage
### Load Dataset
Modify `bert_exercise_dmoz_health.ipynb` with your dataset file name and column names:
```python
# Define parameters
data_file = "Dmoz-Health.csv"
text_column = "text"
label_column = "class"
```

### Run the Notebook
Open `bert_training.ipynb` and execute all cells.
```sh
jupyter notebook
```

## Results
After running the training, the evaluation metrics (Accuracy, F1-score, and Confusion Matrix) will be displayed.

## Features
✅ **Supports Custom Datasets** (Specify CSV file & column names)  
✅ **Modular Codebase** (Separate functions for training, tokenization, etc.)  
✅ **BERT Model Integration** (Pretrained `bert-base-uncased`)  
✅ **Train/Test Split (70/10/20)**