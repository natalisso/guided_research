from dataclasses import dataclass

@dataclass
class SuperGLUETaskConfigs:
    """
    Class for getting task specific training cofigs
    from the SuperGLUE benchmark.
    """
    task_name : str
    feature_keys : list[str]
    label_key : str
    label_names : list[str] | None
    evaluation_metric : str

SUPER_GLUE_DATASETS_INFOS : dict[str, SuperGLUETaskConfigs] = {
  "axb": SuperGLUETaskConfigs(
    task_name="axb",
    feature_keys=["sentence1", "sentence2"],
    label_key="label",
    label_names=["not_entailment", "entailment"],
    evaluation_metric="matthews_correlation"),
  "axg": SuperGLUETaskConfigs(
    task_name="axg",
    feature_keys=["premise", "hypothesis"],
    label_key="label",
    label_names=["not_entailment", "entailment"],
    evaluation_metric="acc_and_f1"),
  "boolq": SuperGLUETaskConfigs(
    task_name="boolq",
    feature_keys=["passage", "question"],
    label_key="label",
    label_names=["False", "True"],
    evaluation_metric="accuracy"),
  "cb": SuperGLUETaskConfigs(
    task_name="cb",
    feature_keys=["premise", "hypothesis"],
    label_key="label",
    label_names=["entailment", "contradiction", "neutral"],
    evaluation_metric="acc_and_f1"),
  "copa": SuperGLUETaskConfigs(
    task_name="copa",
    feature_keys=["premise", "choice1", "choice2", "question"],
    label_key="label",
    label_names=["choice1", "choice2"],
    evaluation_metric="accuracy"),
  "multirc": SuperGLUETaskConfigs(
    task_name="multirc",
    feature_keys=["paragraph", "question", "answer"],
    label_key="label",
    label_names=["False", "True"],
    evaluation_metric="acc_and_f1"),
  "record": SuperGLUETaskConfigs(
    task_name="record",
    feature_keys=["passage", "query"],
    label_key="answers",
    label_names=None,
    evaluation_metric="acc_and_f1"),
  "rte": SuperGLUETaskConfigs(
    task_name="rte",
    feature_keys=["premise", "hypothesis"],
    label_key="label",
    label_names=["entailment", "not_entailment"],
    evaluation_metric="accuracy"),
  "wic": SuperGLUETaskConfigs(
    task_name="wic",
    feature_keys=["sentence1", "sentence2", "word"],
    label_key="label",
    label_names=["False", "True"],
    evaluation_metric="accuracy"),
  "wsc": SuperGLUETaskConfigs(
    task_name="wsc.fixed",
    feature_keys=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
    label_key="label",
    label_names=["False", "True"],
    evaluation_metric="accuracy"),
}