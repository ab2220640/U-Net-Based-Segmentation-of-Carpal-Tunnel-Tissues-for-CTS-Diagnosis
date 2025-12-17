import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm  # 導入進度條套件
import zipfile  # [新增] 用於自動解壓縮

# ==========================================
# 設定裝置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 損失函數 (Dice Loss)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
 
    def forward(self, input, target, weights=None):
        C = target.shape[1]
        dice = DiceLoss()
        totalLoss = 0
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        return totalLoss / C

# ==========================================
# 2. U-Net 模型架構 (保持不變)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = DoubleConv(512, 1024 // factor)
        self.pool = nn.MaxPool2d(2)
        self.up1 = DoubleConv(1024, 512 // factor)
        self.up2 = DoubleConv(512, 256 // factor)
        self.up3 = DoubleConv(256, 128 // factor)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        
        x = self.up_sample(x5)
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1)
        x = self.up1(x)

        x = self.up_sample(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.up2(x)

        x = self.up_sample(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.up3(x)

        x = self.up_sample(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.up4(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)

# ==========================================
# 3. 資料集定義
# ==========================================
class CarpalTunnelDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = []
        
        # 檢查 root_dir 是否存在，避免報錯
        if not os.path.exists(root_dir):
            print(f"警告：找不到資料夾 {root_dir}")
            return

        for i in range(10): # 0~9 資料夾
            case_path = os.path.join(root_dir, str(i))
            # 搜尋圖片，支援 jpg 或 png
            t1_files = sorted(glob.glob(os.path.join(case_path, 'T1', '*.*')))
            
            for t1_path in t1_files:
                filename = os.path.basename(t1_path)
                t2_path = os.path.join(case_path, 'T2', filename)
                mn_path = os.path.join(case_path, 'MN', filename)
                ft_path = os.path.join(case_path, 'FT', filename)
                ct_path = os.path.join(case_path, 'CT', filename)
                
                if os.path.exists(t2_path):
                    self.data_list.append({
                        't1': t1_path, 't2': t2_path,
                        'mn': mn_path, 'ft': ft_path, 'ct': ct_path
                    })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        t1_img = Image.open(item['t1']).convert('L')
        t2_img = Image.open(item['t2']).convert('L')
        
        def read_mask(path):
            if os.path.exists(path):
                return Image.open(path).convert('L')
            else:
                return Image.new('L', t1_img.size, 0)
        
        mn_mask = read_mask(item['mn'])
        ft_mask = read_mask(item['ft'])
        ct_mask = read_mask(item['ct'])
        
        to_tensor = transforms.ToTensor()
        
        # 組合影像 (2 channels)
        image = torch.cat((to_tensor(t1_img), to_tensor(t2_img)), dim=0)
        # 組合 Mask (3 channels: MN, FT, CT)
        mask = torch.cat((to_tensor(mn_mask), to_tensor(ft_mask), to_tensor(ct_mask)), dim=0)
        mask = (mask > 0.5).float() # 二值化

        return image, mask

# ==========================================
# 4. 訓練與輔助功能
# ==========================================
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, epoch_idx):
    loop = tqdm(loader, desc=f"Epoch {epoch_idx}", leave=False)
    model.train()
    epoch_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return epoch_loss / len(loader)

def check_accuracy(loader, model, device="cuda"):
    dice_loss_fn = MulticlassDiceLoss()
    model.eval()
    dice_score = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float()
            
            loss = dice_loss_fn(preds, y)
            dice_score += (1 - loss.item())
            num_batches += 1
            
    model.train()
    if num_batches == 0: return 0
    return dice_score / num_batches

# ==========================================
# Main 函式 (包含自動路徑偵測)
# ==========================================
def main():
    print(f" 系統偵測到: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------
    # [新增] 自動路徑設定 & 解壓縮邏輯
    # ----------------------------------------------------
    # 取得 train.py 所在的絕對路徑
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 定義資料夾與檔案路徑
    DATA_DIR = os.path.join(CURRENT_DIR, 'dataset')
    ZIP_FILE = os.path.join(CURRENT_DIR, 'dataset.zip')
    CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')

    # 1. 確保 Checkpoint 資料夾存在
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f" 已建立模型儲存資料夾: {CHECKPOINT_DIR}")

    # 2. 自動解壓縮 Dataset (如果資料夾不存在但 zip 存在)
    if not os.path.exists(DATA_DIR):
        if os.path.exists(ZIP_FILE):
            print(" 偵測到壓縮檔，正在自動解壓縮 dataset.zip ...")
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(CURRENT_DIR)
            print(" 解壓縮完成！")
        else:
            print(f" 錯誤：找不到 dataset 資料夾，也找不到 dataset.zip。路徑: {DATA_DIR}")
            # 如果真的沒資料，程式還是會往下跑但 Dataset len 會是 0
    # ----------------------------------------------------

    print("開始準備訓練流程...")

    # --- 參數設定 ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4      
    NUM_EPOCHS = 50     
    NUM_WORKERS = 2     
    NUM_FOLDS = 5       
    
    # 載入資料集 (使用自動偵測的路徑)
    full_dataset = CarpalTunnelDataset(DATA_DIR)
    print(f"總共載入 {len(full_dataset)} 組影像資料")

    if len(full_dataset) == 0:
        print("警告：沒有讀取到任何資料，請檢查 dataset 資料夾結構！")
        return
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # 讀取訓練進度 (中斷續練功能)，紀錄檔也放在 checkpoints 裡比較整齊
    start_fold = 0
    fold_record_file = os.path.join(CHECKPOINT_DIR, "fold_status.txt")
    
    if os.path.exists(fold_record_file):
        with open(fold_record_file, "r") as f:
            content = f.read().strip()
            if content:
                start_fold = int(content)
        print(f"偵測到上次進度，將從 Fold {start_fold + 1} 繼續訓練...")

    # --- Fold 迴圈 ---
    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
        if fold < start_fold:
            continue
            
        print(f'\n========================================')
        print(f'現在開始 Fold {fold+1}/{NUM_FOLDS}')
        print(f'========================================')
        
        train_subsampler = Subset(full_dataset, train_ids)
        test_subsampler = Subset(full_dataset, test_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_subsampler, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        model = UNet(n_channels=2, n_classes=3).to(device)
        loss_fn = MulticlassDiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()

        # [修改] 讀取該 Fold 的暫存檔 (路徑指向 checkpoints)
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_fold_{fold}.pth.tar")
        
        start_epoch = 0
        best_val_dice = 0
        
        if os.path.exists(checkpoint_file):
            print(f"載入 Fold {fold+1} 的存檔點...")
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_val_dice = checkpoint.get('best_dice', 0)

        # --- Epoch 迴圈 ---
        for epoch in range(start_epoch, NUM_EPOCHS):
            avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, epoch+1)
            val_dice = check_accuracy(test_loader, model, device=device)
            
            tqdm.write(f"Fold {fold+1} | Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f}")

            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_dice': best_val_dice
            }
            save_checkpoint(checkpoint, filename=checkpoint_file)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                # [修改] 最佳模型存檔路徑 (指向 checkpoints)
                best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), best_model_path)
                tqdm.write(f"💾 發現更佳模型！已儲存至 {best_model_path} (Dice: {best_val_dice:.4f})")
        
        # 該 Fold 完成，更新進度紀錄
        with open(fold_record_file, "w") as f:
            f.write(str(fold + 1))
            
        print(f"Fold {fold+1} 訓練結束！最佳 Dice Score: {best_val_dice:.4f}")

    print("\n全數訓練完成！檔案已生成。")

if __name__ == "__main__":
    main()

