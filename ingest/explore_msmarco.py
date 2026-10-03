from datasets import load_dataset

ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
print(ds)
print(ds.features)

row = ds[0]
print("QUERY:", row["query"])
print("ANSWERS:", row["answers"])
print("PASSAGE KEYS:", row["passages"].keys())
print("NUM PASSAGES:", len(row["passages"]["passage_text"]))
print("IS_SELECTED:", row["passages"]["is_selected"])
print("URL:", row["passages"]["url"][0])
print("FIRST PASSAGE:", row["passages"]["passage_text"][0][:300])

# how many queries are usable as eval queries?
usable = [r for r in ds
          if r["answers"] and r["answers"][0] != "No Answer Present."
          and sum(r["passages"]["is_selected"]) > 0]
print("Queries with an answer and a labelled relevant passage:", len(usable))