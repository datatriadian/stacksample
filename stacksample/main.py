from stacksample.console import console
from stacksample.loader import load_answers, load_questions, load_tags


def main() -> int:
    with console.status("Loading data..."):
        answers = load_answers()
        questions = load_questions()
        tags = load_tags()

    console.print("Answers sample\n")
    console.print(answers.head())
    console.print("\n")
    console.print("Questions sample\n")
    console.print(questions.head())
    console.print("\n")
    console.print("Tags sample\n")
    console.print(tags.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
