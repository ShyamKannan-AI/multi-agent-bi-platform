from pprint import pprint

from src.ingestion import load_dataset
from src.profiling import profile_dataset


def main():

    file_path = "C:/Users/T9935/Downloads/retail_sales_dataset.csv"

    df = load_dataset(file_path)

    profile = profile_dataset(df)

    print("\n==============================")
    print("DATASET PROFILE")
    print("==============================")

    print(f"\nRows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")

    print("\nColumn Details:")

    for column, details in profile["column_details"].items():

        print(f"\n--- {column} ---")

        print(f"Data type: {details['dtype']}")
        print(f"Missing: {details['missing_count']}")
        print(f"Missing %: {details['missing_percentage']}")
        print(f"Unique values: {details['unique_count']}")

        if "statistics" in details:
            print("\nStatistics:")
            pprint(details["statistics"])

        if "top_values" in details:
            print("\nTop values:")
            pprint(details["top_values"])


if __name__ == "__main__":
    main()