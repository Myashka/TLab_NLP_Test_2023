from datasets import load_dataset
from datasets.combine import concatenate_datasets

def make_datasets(
    dataset_name,
    test_size=1000,
    val_size=500,
):

    dataset = load_dataset(dataset_name)["train"]
    
    train_test_split = dataset.train_test_split(test_size=test_size)
    temp_train_data = train_test_split["train"]
    test_data = train_test_split["test"]

    train_val_split = temp_train_data.train_test_split(test_size=val_size)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]  

    col_to_remove = [col for col in train_data.column_names if col not in ['prompt', 'chosen', 'rejected']]
    train_data = train_data.remove_columns(col_to_remove)
    val_data = val_data.remove_columns(col_to_remove)

    return train_data, val_data