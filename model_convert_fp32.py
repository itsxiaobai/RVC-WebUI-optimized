import torch
import os

# 1. 讓使用者輸入模型名稱
base_name = input("請輸入模型名稱 (不含副檔名): ").strip()

# 2. 設定路徑
path = f"assets/weights/{base_name}.pth"

# 檢查檔案是否存在
if not os.path.exists(path):
    print(f"\n❌ 錯誤: 找不到檔案 {path}")
    print("請確認檔案存在於 assets/weights/ 目錄下")
    exit(1)

print(f"\n📂 載入路徑: {path}")

# 3. 載入模型權重
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
except Exception as e:
    print(f"\n❌ 載入失敗: {e}")
    exit(1)

# 4. 顯示原始權重的數據類型
print("\nOriginal weight dtypes (前 5 個權重):")
if isinstance(ckpt, dict) and "weight" in ckpt:
    weight_dict = ckpt["weight"]
else:
    print("⚠️ 注意: 'weight' 鍵值不存在，假設 ckpt 本身為 state_dict")
    weight_dict = ckpt

for key in list(weight_dict.keys())[:5]:
    print(f"  {key}: {weight_dict[key].dtype}")

# 5. 轉換 float16 → float32
converted_count = 0
total_count = len(weight_dict)

for key in weight_dict:
    if weight_dict[key].dtype == torch.float16:
        weight_dict[key] = weight_dict[key].float()
        converted_count += 1

print(f"\n✅ 轉換完成: {converted_count}/{total_count} 個權重從 float16 轉為 float32")

# 6. 儲存新檔案
new_path = path.replace(".pth", "_fp32.pth")
torch.save(ckpt, new_path)
print(f"\n💾 已儲存: {new_path}")
