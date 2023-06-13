import datasets as ds

dataset = ds.load_from_disk("test")
for row in dataset:
    print(row)