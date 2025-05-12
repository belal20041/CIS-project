@echo off
echo Starting batch file...

REM Create directories for project structure
mkdir data deployment docs mlops models notebooks pages tests utils

REM Create placeholder files
echo. > deployment\Dockerfile
echo. > deployment\api.py
echo. > deployment\requirements.txt
echo. > docs\final_report.md
echo. > docs\slides_template.pptx
echo. > mlops\tracker.py
echo. > mlops\versioner.py
echo. > models\evaluator.py
echo. > models\persistence.py
echo. > models\trainer.py
echo. > pages\milestone1.py
echo. > pages\milestone2.py
echo. > pages\milestone3.py
echo. > pages\milestone4.py
echo. > pages\milestone5.py
echo. > tests\test_api.py
echo. > tests\test_data.py
echo. > tests\test_model.py
echo. > utils\data_loader.py
echo. > utils\preprocessor.py
echo. > utils\visualizer.py
echo Done creating placeholder files!

REM Add data files to DVC
echo Adding files to DVC...
dvc add data\Submission.csv
dvc add data\submission_xgboost.csv
dvc add data\Test.csv
dvc add data\test_processed.parquet
dvc add data\Train.csv
dvc add data\train_processed.parquet
dvc add data\val_processed.parquet

REM Add all files to Git
echo Committing to Git...
git add .
git commit -m "Add placeholder files and DVC metadata"

REM Push to Git remote
echo Pushing to Git...
git push origin main

REM Push DVC-tracked data to remote
echo Pushing to DVC remote...
dvc push

echo Batch file completed successfully!