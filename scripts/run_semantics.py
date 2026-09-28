from pprint import pprint

from src.ingestion import load_dataset
from src.profiling import profile_dataset
from src.semantics import SemanticAnalyzer


def main():

    file_path = "C:/Users/T9935/Downloads/retail_sales_dataset.csv"

    print("\nLoading dataset...")

    df = load_dataset(file_path)

    print("Generating profile...")

    profile = profile_dataset(df)

    print("Running semantic analysis...")

    analyzer = SemanticAnalyzer()

    semantic_context = analyzer.analyze(
        df,
        profile,
    )

    print("\n==============================")
    print("SEMANTIC DATASET ANALYSIS")
    print("==============================")

    pprint(semantic_context)


if __name__ == "__main__":
    main()