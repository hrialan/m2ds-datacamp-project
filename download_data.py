import zipfile
 
compressed_dataset_path = "./dataset/airbnb_compressed.zip"

def extract_file(zip_file_path):
    try:
        with zipfile.ZipFile(zip_file_path) as z:
            z.extractall("./dataset")
            print("Files were successfully extracted")
    except:
        print("Invalid file")


if __name__ == "__main__": 
    extract_file(compressed_dataset_path)
