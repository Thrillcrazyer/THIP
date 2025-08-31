import pandas as pd


def get_dataset(dataset:pd.DataFrame,log_path="eventlogs/DeepMath_eventlog.csv")->pd.DataFrame:
    log=pd.read_csv(log_path)
    case_ids = log["Case ID"].unique()
    filtered_dataset = dataset.loc[case_ids]
    return filtered_dataset

if __name__ == "__main__":
    dataset=pd.read_csv("DeepMath-103k.csv")
    df=get_dataset(dataset)
    breakpoint()