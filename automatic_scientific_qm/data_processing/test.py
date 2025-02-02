from automatic_scientific_qm.section_classification.trainer import SectionClassifier

section_classifier = SectionClassifier.load_from_checkpoint(
    "section_classifier_openreview.ckpt",
    config={
        "model_type": "transformer",
        "lr": 0.0001,
        "model": {"num_classes": 5, "num_layers": 2, "dropout": 0.0},
        "load": {"load": False, "experiment_name": "", "run_id": "", "checkpoint": ""},
    },
    map_location="cuda:0",
)


section_classifier.save_checkpoint("section_classifier_openreview2.ckpt")
