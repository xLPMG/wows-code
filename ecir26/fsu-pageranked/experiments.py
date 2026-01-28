import pyterrier as pt
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from ir_measures import nDCG

print("1")
ensure_pyterrier_is_loaded()
ds_id = "radboud-validation-20251114-training"
dataset = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{ds_id}")
topics = dataset.get_topics("title")

qrels = dataset.get_qrels()

print(topics)
print(f"{len(qrels)} qrels loaded:")
print(qrels)
exit()
print("2")
ows_bm_25 = pt.Artifact.from_url("tira:radboud-validation-20251114-training/ows/pyterrier-BM25-on-title")
# golden_retrieval = pt.Artifact.from_url("tira:radboud-validation-20251114-training/ks-golden-retrievals/pyterrier-on-default_text-with-DPH-Bo1-DPH")
# ows_pl2 = pt.Artifact.from_url("tira:radboud-validation-20251114-training/ows/pyterrier-PL2-on-title")
# orakel_monot5 = pt.Artifact.from_url("tira:radboud-validation-20251114-training/ks-orakel/bm25 + monoT5 reranker")
# chatnoir_desc = pt.Artifact.from_url("tira:radboud-validation-20251114-training/ows/chatnoir-description-default-10")
# chatnoir_title = pt.Artifact.from_url("tira:radboud-validation-20251114-training/ows/chatnoir-title-bm25-100")
fsu_pageranked = pt.Artifact.from_url("tira:radboud-validation-20251114-training/fsu-pageranked/fsu-pageranked-bm25-all-fields-bo1-qe")

print("4")
fsu_pageranked(topics)
print("5")

results = pt.Experiment(
    [fsu_pageranked, ows_bm_25], # weighting techniques
    topics,
    qrels,
    [nDCG(judged_only=True)@10],
    names=["FSU Pageranked", "BM25 (OWS)"],
    baseline=1, # ID of baseline
    test="t", # test to use; here: Student's t-test
    correction="bonferroni" # correction for multiple testing
)

print(results)