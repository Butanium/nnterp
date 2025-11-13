from nnterp.rename_utils import get_rename_dict

rename = get_rename_dict(rename_config=None)
print("Rename dict for GPT-2:")
for k, v in sorted(rename.items()):
    print(f"  {k!r}: {v!r}")
