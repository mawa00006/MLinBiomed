import pandas as pd


def return_df(file):
    """Reads the uploaded file based on its extension and returns a pandas DataFrame."""
    name = file.name
    extension = name.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(file)
    elif extension == "tsv":
        df = pd.read_csv(file, sep="\t")
    elif extension == "xlsx":
        df = pd.read_excel(file)
    elif extension == "xml":
        df = pd.read_xml(file)
    elif extension == "json":
        df = pd.read_json(file)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    return df