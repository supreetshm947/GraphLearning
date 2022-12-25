import os
import shutil
import pandas as pd
import tarfile
import numpy as np

ARCHIVE_NAME = "cora"

tar_file = tarfile.open(ARCHIVE_NAME + ".tgz", "r")
try:
    tar_file.extractall()
    citations = pd.read_csv(
        os.path.join(ARCHIVE_NAME, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )
    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(ARCHIVE_NAME, "cora.content"), sep="\t", header=None, names=column_names,
    )
finally:
    if os.path.exists(ARCHIVE_NAME):
        shutil.rmtree(ARCHIVE_NAME)

# converting categorical val to integer

class_idx = {name: id for id, name in enumerate(sorted(papers.subject.unique()))}
paper_idx = {name: id for id, name in enumerate(sorted(papers.paper_id.unique()))}

papers.paper_id = papers.paper_id.apply(lambda name: paper_idx[name])
citations.source = citations.source.apply(lambda name: paper_idx[name])
citations.target = citations.target.apply(lambda name: paper_idx[name])
papers.subject = papers.subject.apply(lambda name: class_idx[name])

#splitting data to train/test

train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

print(train_data.shape)
print(test_data.shape)