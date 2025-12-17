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
import zipfile  # 用於自動解壓縮

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
# 2. U-Net 模型架構
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
        
        if not os.path.exists(root_dir):
            print(f"警告：找不到資料夾 {root_dir}，Dataset 為空")
            return

        for i in range(10): # 0~9 資料夾
            case_path = os.path.join(root_dir, str(i))
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
        
        image = torch.cat((to_tensor(t1_img), to_tensor(t2_img)), dim=0)
        mask = torch.cat((to_tensor(mn_mask), to_tensor(ft_mask), to_tensor(ct_mask)), dim=0)
        mask = (mask > 0.5).float()

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
# 輔助：智慧尋找資料集路徑 [新增功能]
# ==========================================
def find_and_prepare_dataset(current_script_path):
    """
    搜尋 dataset 資料夾或 zip 檔。
    搜尋順序:
    1. 專案根目錄 (src 的上一層)
    2. src 資料夾 (程式所在目錄)
    """
    src_dir = os.path.dirname(os.path.abspath(current_script_path))
    project_root = os.path.dirname(src_dir)
    
    # 定義要搜尋的路徑列表 (優先搜尋根目錄，因為通常 dataset 會放在專案最外層)
    search_paths = [project_root, src_dir]
    
    # 1. 先找有沒有已經解壓縮的 'dataset' 資料夾
    for path in search_paths:
        target_dir = os.path.join(path, 'dataset')
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            print(f"發現資料集資料夾: {target_dir}")
            return target_dir

    # 2. 如果沒找到資料夾，找 'dataset.zip' 並解壓縮
    for path in search_paths:
        zip_file = os.path.join(path, 'dataset.zip')
        if os.path.exists(zip_file):
            print(f"發現壓縮檔: {zip_file}，正在解壓縮...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(path) # 解壓到該 zip 所在的資料夾
            print(f"解壓縮完成！")
            return os.path.join(path, 'dataset')

    # 3. 都找不到，拋出錯誤
    raise FileNotFoundError(
        f"找不到 'dataset' 資料夾或 'dataset.zip'。\n"
        f"請確保檔案位於專案根目錄 ({project_root}) 或 src 資料夾 ({src_dir}) 內。"
    )

# ==========================================
# Main 函式
# ==========================================
def main():
    print(f"系統偵測到: {torch.cuda.get_device_name(0)}")

    # 1. 智慧尋找資料集路徑
    try:
        DATA_DIR = find_and_prepare_dataset(__file__)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. 設定 checkpoints 儲存位置 (存放在 src/checkpoints，保持專案整潔)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(SRC_DIR, '../checkpoints') # 存到專案根目錄的 checkpoints
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"已建立模型儲存資料夾: {CHECKPOINT_DIR}")

    print("開始準備訓練流程...")

    # --- 參數設定 ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4      
    NUM_EPOCHS = 50     
    NUM_WORKERS = 2     
    NUM_FOLDS = 5       
    
    # 載入資料集
    full_dataset = CarpalTunnelDataset(DATA_DIR)
    print(f"總共載入 {len(full_dataset)} 組影像資料")
    
    if len(full_dataset) == 0:
        print("錯誤：資料夾內沒有影像，請檢查資料結構。")
        return

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # 讀取訓練進度
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

        # 讀取 Checkpoint
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

            # 儲存暫存檔
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_dice': best_val_dice
            }
            save_checkpoint(checkpoint, filename=checkpoint_file)

            # 儲存最佳模型
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), best_model_path)
                tqdm.write(f"發現更佳模型！已儲存至 {best_model_path} (Dice: {best_val_dice:.4f})")
        
        # 更新進度紀錄
        with open(fold_record_file, "w") as f:
            f.write(str(fold + 1))
            
        print(f"Fold {fold+1} 訓練結束！最佳 Dice Score: {best_val_dice:.4f}")

    print("\n全數訓練完成！檔案已生成。")

if __name__ == "__main__":
    main()