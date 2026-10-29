import pandas as pd

from src.make_demo_sample import sample_dataset


def test_sample_dataset_strips_label_header_and_preserves_columns(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "sample.csv"
    pd.DataFrame(
        [[1, "BENIGN"], [2, "Attack"], [3, "Attack"]],
        columns=[" Feature ", " Label"],
    ).to_csv(source, index=False)

    metadata = sample_dataset(
        source=source,
        output=output,
        samples_per_label=10,
        chunk_size=2,
        per_chunk_per_label=10,
    )

    result = pd.read_csv(output)
    assert list(result.columns) == ["Feature", "Label"]
    assert metadata["label_distribution"] == {"Attack": 2, "BENIGN": 1}
    assert output.read_bytes()
