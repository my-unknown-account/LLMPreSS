"""Interactive search; requires an index download on first use."""

from llmpress import SemanticIndex


def main():
    query = input("Query: ").strip()
    if not query:
        return
    top_k = int(input("Number of results [10]: ").strip() or "10")
    index = SemanticIndex("ClimbMix", edition="compact", index_dir=None)
    for result in index.search(query, top_k=top_k):
        print(f"\n{result.rank}. {result.passage_id} (score={result.score:.4f})")
        print(result.text)


if __name__ == "__main__":
    main()
