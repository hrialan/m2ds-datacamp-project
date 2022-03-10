import zipfile
from zipfile import BadZipFile

compressed_dataset_path = "./data/airbnb_compressed.zip"


def extract_file(zip_file_path):
    try:
        with zipfile.ZipFile(zip_file_path) as z:
            z.extractall("./data")
            print("Files were successfully extracted")
    except BadZipFile:
        print("Invalid file")


if __name__ == "__main__":
    extract_file(compressed_dataset_path)
