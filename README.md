# Trip Purpose Inference ML Codebase

This repository accompanies the academic paper **"Navigating Machine Learning for Travel Behavior Analysis: A Comprehensive Guide to Inferring Trip Purposes Using Transit Survey and Automatic Fare Collection Data Fusion,"** currently under review. It contains the data preparation, feature engineering, and model tuning pipelines used to reproduce the experiments described in the manuscript.

## Repository Structure

```         
.
├── data/
│   ├── dataIn.csv           # Primary input: 2022 OBS
│   ├── study1keys.csv       # Feature engineering round 1 (not in the paper)
│   └── study2keys.csv       # Feature engineering round 2 (Ch. 6, Stage 1)
├── outputs/
│   ├── optunaStudies.db     # Optuna storage containing hyperparameter trials
│   ├── *Tuning.csv          # Saved Optuna study histories for each ML model
│   ├── Final_*best.csv      # Best-performing configurations per model 
│   ├── Final_*worst.csv     # Worst-performing configurations per model
│   └── Round1/              # Archived artifacts from preliminary experimentation
├── src/
│   ├── preprocessing.py     # Data wrangling and encoding options
│   └── mlmodels.py          # Model training wrappers (RF, XG, SVM, CB, NN)
├── main.py                  # Interactive Optuna-driven tuning workflow
├── LICENSE                  # Licensing information
├── UnderReview.pdf          # The submitted paper (under review)
└── README.md                # Project overview (this file)
```

## Code Overview

-   **`main.py`**
    -   Sets the working directory and loads the fused dataset from `data/dataIn.csv`.
    -   Calls `src.preprocessing.getVars` to assemble feature engineering configurations and train/test splits.
    -   Prompts the analyst to select a target model family and batch size, optionally enabling GPU execution for gradient-boosted models.
    -   Defines Optuna objective functions (`objective_RF`, `objective_XG`, `objective_SV`, `objective_NN`, `objective_CB`) that wrap the training helpers in `src/mlmodels.py`.
    -   Launches hyperparameter optimization runs and logs results to both CSV summaries and the Optuna SQLite store under `outputs/`.
-   **`src/preprocessing.py`**
    -   Declares reusable encoding dictionaries (`OPTDICT`) for categorical feature level consolidation.
    -   Provides `getVars` to generate all combinations of feature engineering strategies and align them with model choices for sequential Optuna sweeps.
    -   Implements a suite of feature builders (e.g., cyclic time encodings, positional encodings, polar/elliptic coordinate transforms) used during experimentation.
    -   Establishes deterministic random seeds for reproducibility across study rounds.
-   **`src/mlmodels.py`** *(inspect this file for implementation details specific to each learner)*
    -   Houses wrappers around scikit-learn, XGBoost, CatBoost, and PyTorch models.
    -   Standardizes model training, evaluation metric computation (accuracy, class-wise F1), and return signatures expected by the Optuna objectives.
-   **`data/`**
    -   `dataIn.csv`: Stores the Metropolitan Council's 2022 Transit On-boar survey where zone memberships are precomputed.
    -   Also stores auxiliary lookup tables required for reproducing the feature engineering configurations (round 1: preliminary; round 2: described in the paper); i.e., it ships with placeholder CSVs. They are generated in `if name=='main'` block of `src/preprocessing.py` from the object `dfCombinations`.
-   **`outputs/`**
    -   Collects Optuna artifacts, enabling post-hoc analysis of hyperparameter search performance, best/worst trials, and reproducing tables/figures for the manuscript.

## Setup Requirements

1.  **Python environment**
    -   Python 3.9+ is recommended.

    -   Create a virtual environment and install dependencies:

        ``` bash
        python -m venv .venv
        source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
        pip install -r requirements.txt
        ```

        *If `requirements.txt` is not provided, manually install the libraries referenced in `src/mlmodels.py` and `src/preprocessing.py` (e.g., numpy, pandas, scikit-learn, optuna, xgboost, catboost, torch, haversine).*
2.  **Data placement**
    -   Place the dataset and key tables in the `data/` directory using the exact filenames shown above. The Optuna routines expect `data/dataIn.csv` to exist.
3.  **GPU acceleration (optional)**
    -   For CUDA-capable GPUs can be enabled via the runtime prompt when launching `main.py`. Ensure the appropriate GPU drivers and Python packages (`xgboost`, `catboost`) are installed with GPU support.
    -   It requires additional setup, but the code also supports the use of `rapids::cudf` for CUML compatible machines (requires WSL2 or Linux AND NVIDIA GPU with 7+); user can hardcode the condition to trigger this feature by changing the line `if os.name=='posix' and os.getenv('USER')=='sh222'` in **`src/mlmodels.py`**.

## Running Experiments

1.  Activate your Python environment and ensure the working directory is the repository root.

2.  Launch the interactive tuning workflow:

    ``` bash
    python main.py
    ```

3.  Follow the console prompts to select a model family, batching strategy, and GPU usage. Optuna study results will be saved automatically in `outputs/`.

## Reproducibility Notes

-   Random seeds are fixed in `src/preprocessing.py` to stabilize feature generation and data splits across sessions.
-   Optuna study histories are stored in `outputs/optunaStudies.db`; you can resume or analyze past runs by reusing this database.
-   Best and worst trial summaries in `outputs/Final_*{best,worst}.csv` mirror the configurations discussed in the paper, facilitating direct comparison with published results.

For questions or collaboration requests related to the paper, please refer to the contact information in the manuscript.
