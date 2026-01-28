#!/usr/bin/env python3
import click
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm

def extract_text_of_document(doc):
    parts = []
    if doc.title:
        parts.append(doc.title * 3) # Weigh title 3 times more
    if doc.description:
        parts.append(doc.description * 2) # Weigh description 2 times more
    if doc.default_text():
        parts.append(doc.default_text())
    return "\n".join(parts)


def get_index(dataset, field, output_path):
    print(f"Specific extraction for field: {field} ignored, using all fields combined.")
    index_dir = output_path / "indexes" / f"{dataset}-on-all-fields"
    if not index_dir.is_dir():
        print("Build new index")
        docs = []
        dataset = ir_datasets.load(f"ir-lab-wise-2025/{dataset}")

        for doc in tqdm(dataset.docs_iter(), "Pre-Process Documents"):
            docs.append({"docno": doc.doc_id, "text": extract_text_of_document(doc)})

        with tracking(export_file_path=index_dir / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
            pt.IterDictIndexer(str(index_dir.absolute()), meta={'docno' : 100}, verbose=True).index(docs)

    return pt.IndexFactory.of(str(index_dir.absolute()))


def run_retrieval(output, index, dataset, retrieval_model, text_field_to_retrieve, run_experiment_flag):
    tag = f"pyterrier-{retrieval_model}-with-query-expansion-on-all-fields"
    target_dir = output / "runs" / dataset / tag
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        return

    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset}").get_topics("title")

    retriever = pt.terrier.Retriever(index, wmodel=retrieval_model, controls={"qemodel" : "Bo1", "qe" : "on"}) # Using Bo1 query expansion

    description = f"This is a PyTerrier retriever using the retrieval model {retriever} retrieving on all text fields (title, description and text) of the text representation of the documents. It uses Bo1 query expansion."

    with tracking(export_file_path=target_dir / "ir-metadata.yml", export_format=ExportFormat.IR_METADATA, system_description=description, system_name=tag):
        run = retriever(topics)

    pt.io.write_results(run, target_file)

    if run_experiment_flag:
        print("Running experiment...")
        run_experiment(retriever, dataset)

def run_experiment(retriever, dataset_id):
    ds = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset_id}")

    topics = ds.get_topics("title")
    qrels = ds.get_qrels()

    ex = pt.Experiment(
        [retriever],
        topics,
        qrels,
        eval_metrics=["ndcg_cut_5", "ndcg_cut_10"]
    )

    print(ex)

@click.command()
@click.option("--dataset", type=click.Choice(["radboud-validation-20251114-training", "spot-check-20251122-training"]), required=True, help="The dataset.")
@click.option("--output", type=Path, required=False, default=Path("output"), help="The output directory.")
@click.option("--retrieval-model", type=str, default="BM25", required=False, help="The retrieval model (e.g., BM25, PL2, DirichletLM).")
@click.option("--text-field-to-retrieve", type=click.Choice(["default_text", "title", "description"]), required=False, default="default_text", help="The text field of the documents on which to retrieve.")
@click.option("--experiment", is_flag=True, default=False, help="Whether to run the experiment after retrieval.")
def main(dataset, text_field_to_retrieve, retrieval_model, output, experiment):
    ensure_pyterrier_is_loaded(is_offline=False)

    index = get_index(dataset, text_field_to_retrieve, output)
    run_retrieval(output, index, dataset, retrieval_model, text_field_to_retrieve, run_experiment_flag=experiment)
    

if __name__ == '__main__':
    main()
