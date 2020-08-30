import optuna

study_path = "sqlite:///yolov3_study.db"
output_path = "./tmp.csv"

study = optuna.load_study(study_name="yolov3_study", storage=study_path)

results = study.trials_dataframe()
results.to_csv(output_path)