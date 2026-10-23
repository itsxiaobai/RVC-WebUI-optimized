import torch

path = "assets/weights/Xe.pth"
ckpt = torch.load(path, map_location="cpu")

print("Original weight dtypes:")
for key in list(ckpt["weight"].keys())[:5]:
    print(f"  {key}: {ckpt['weight'][key].dtype}")

# 全部轉成 fp32
for key in ckpt["weight"]:
    if ckpt["weight"][key].dtype == torch.float16:
        ckpt["weight"][key] = ckpt["weight"][key].float()

# 另存新檔
new_path = path.replace(".pth", "_fp32.pth")
torch.save(ckpt, new_path)
print(f"\nSaved as: {new_path}")
