import pandas as pd

def load_data(path):
    try:
        return pd.read_csv(path, delimiter=',', quoting=1, on_bad_lines='warn')
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

for SEC in [1,2,3,5]:
    eval01_1 = load_data(f"./{str(SEC)}sec/evaluation_01_1@{str(SEC)}.csv")
    eval01_2 = load_data(f"./{str(SEC)}sec/evaluation_01_2@{str(SEC)}.csv")
    eval02_1 = load_data(f"./{str(SEC)}sec/evaluation_02_1@{str(SEC)}.csv")
    eval02_2 = load_data(f"./{str(SEC)}sec/evaluation_02_2@{str(SEC)}.csv")
    eval03_1 = load_data(f"./{str(SEC)}sec/evaluation_03_1@{str(SEC)}.csv")
    eval03_2 = load_data(f"./{str(SEC)}sec/evaluation_03_2@{str(SEC)}.csv")

    dfs = [eval01_1, eval01_2, eval02_1, eval02_2, eval03_1, eval03_2]
    first_rows = [df.iloc[0] for df in dfs if df is not None]
    combined_df = pd.DataFrame(first_rows)

    # print("Combined DataFrame:")
    # print(combined_df)

    #csvに保存
    output_path = f"./{str(SEC)}sec/combined_evaluation@{str(SEC)}.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Combined evaluation results saved to: {output_path}")