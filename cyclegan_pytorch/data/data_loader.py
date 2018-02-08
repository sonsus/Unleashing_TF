
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader # under ./data/custom_dataset_data_loader.py
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
