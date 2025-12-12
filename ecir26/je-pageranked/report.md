# Report - je-pageranked Approach

We added the method ``run_experiment`` to evaluate and compare different approaches of `retriever.py` to each other.

## spot-check-20251122-training Dataset

The default `retriever.py` file, which is provided in the repository, achieves a `ndcg_cut_10` score of `0.766625`.

## radboud-validation-20251114-training Dataset

The default `retriever.py` file, which is provided in the repository, achieves a `ndcg_cut_10` score of `0.451635`.

- `extract_text_of_document` title + description + default_text: `0.462461`