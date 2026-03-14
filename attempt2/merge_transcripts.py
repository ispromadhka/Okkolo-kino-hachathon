import pickle, glob
all_transcripts = {}
for f in sorted(glob.glob("new_transcripts_gpu*.pkl")):
    with open(f, "rb") as fh:
        data = pickle.load(fh)
    print(f"{f}: {len(data)} transcripts")
    all_transcripts.update(data)
with open("new_transcripts.pkl", "wb") as f:
    pickle.dump(all_transcripts, f)
print(f"Merged: {len(all_transcripts)} total transcripts")
