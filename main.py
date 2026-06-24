"""
main.py

"""

from src.agent import build_agent_executor


def main():
    print("=" * 60)
    print("FinSight Agent — AI Financial Analyst")
    print("=" * 60)
    print("Ask about a company (e.g. 'investment brief for Apple').")
    print("Ask follow-ups freely -- the agent remembers this session.")
    print("Type 'exit' or 'quit' to stop.\n")

    
    executor = build_agent_executor(verbose=True)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = executor.invoke({"input": user_input})

        print("\n" + "-" * 60)
        print("FinSight Agent:")
        print(result["output"])
        print("-" * 60)


if __name__ == "__main__":
    main()