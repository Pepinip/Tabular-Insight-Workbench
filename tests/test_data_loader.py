import pytest

from src.data_loader import DatasetLoadError, load_tabular_file


def test_load_tabular_file_rejects_unsupported_extension():
    with pytest.raises(DatasetLoadError):
        load_tabular_file("data.txt", b"hello")


def test_load_tabular_file_reads_csv_bytes():
    loaded = load_tabular_file("sample.csv", b"a,b\n1,x\n2,y\n")

    assert loaded.dataframe.shape == (2, 2)
    assert loaded.extension == ".csv"
    assert len(loaded.dataset_hash) == 64