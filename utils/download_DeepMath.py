from datasets import load_dataset
import pandas as pd


def main(output_path: str = "DeepMath-103k.csv", split: str = "train") -> None:
    """DeepMath-103K의 특정 split을 CSV로 저장합니다."""
    # DatasetDict가 아닌 Dataset을 얻기 위해 split을 지정합니다
    ds = load_dataset("zwhe99/DeepMath-103K", split=split)
    # pandas DataFrame으로 변환 후 CSV 저장
    df = ds.to_pandas()
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()