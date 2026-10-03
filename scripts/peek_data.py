from datasets import load_dataset

ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
row = next(iter(ds))

print("KEYS:", list(row.keys()))
print("QUERY:", row["query"])
print("ANSWERS:", row["answers"])
print("IS_SELECTED:", row["passages"]["is_selected"])
print("FIRST PASSAGE:", row["passages"]["passage_text"][0][:200])