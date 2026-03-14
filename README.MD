📈 S&P 500 ETL Project
Study project building a complete ETL pipeline using 50+ years of S&P 500 historical data, ending with a Power BI dashboard.
🛠️ Tech Stack
Python → PostgreSQL → Power BI
StageToolExtractionPython (Pandas)TransformPython (Pandas)LoadPostgreSQLValidationDBeaverVisualizationPower BI
📁 Project Structure
ETL/
├── data/
│ ├── raw_data/ ← original CSV (not versioned)
│ └── processed_data/ ← cleaned CSV
├── scripts/ ← Python ETL code
├── .gitignore
└── README.md
🗃️ Dataset
Download from Kaggle and place the CSV inside data/raw_data/.

⚠️ Data files are not versioned in this repository.

🔍 Data Validation
After each pipeline run, DBeaver is used to inspect the PostgreSQL database — checking tables, data types, row counts, and query results to ensure the load was successful.
📌 Status
🔧 In progress
