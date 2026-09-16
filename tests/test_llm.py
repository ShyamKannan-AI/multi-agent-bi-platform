from src.llm import get_llm


def main():

    llm = get_llm()

    response = llm.invoke(
        "Reply with exactly: LLM connection successful"
    )

    print("\nLLM Response:")
    print(response.content)


if __name__ == "__main__":
    main()