import tkinter as tk #這是 Python 內建做視窗介面的工具
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw #這是用來處理圖片（開啟、縮放、畫圖）的強大工具
import os #用來跟作業系統溝通，比如找檔案路徑
import torch
import torch.nn as nn 
from torchvision import transforms #這是 PyTorch，深度學習的核心。nn 是神經網路模組，transforms 用來把圖片轉成 AI 看得懂的數字格式。
import numpy as np #用來做數學運算，像是把圖片轉成矩陣陣列
import glob #用來搜尋檔案
import time
import threading

# ==========================================
# 1. 模型定義 (U-Net) - 保持不變
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels): #這是 U-Net 的基本積木。它的動作是：卷積 (Conv) -> 標準化 (BatchNorm) -> 活化 (ReLU)，然後重複做兩次。這能幫助 AI 萃取圖片特徵。
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes): #這是模型的「構造圖」
        super(UNet, self).__init__()
        self.bilinear = True
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128) #down1 到 down4 是不斷壓縮圖片，提取深層特徵
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = DoubleConv(512, 1024 // factor)
        self.pool = nn.MaxPool2d(2) #最大池化層，用來把圖片變小（長寬減半）
        self.up1 = DoubleConv(1024, 512 // factor) #up1 到 up4 是把特徵放大，還原回原本圖片的大小
        self.up2 = DoubleConv(512, 256 // factor)
        self.up3 = DoubleConv(256, 128 // factor)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) #最後的輸出層，n_classes 設為 3，代表輸出 3 個通道 (MN, FT, CT)。
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x): #這是資料的「流動方向」。
        x1 = self.inc(x) #資料 x 進來後，先往下走 (down)，然後往上走 (up)。
        x2 = self.down1(self.pool(x1))#這是 U-Net 的左半邊。目標是把圖片「濃縮」，抓出特徵（例如線條、形狀），但圖片尺寸會變小。
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        x = self.up_sample(x5)
        diffY = x4.size()[2] - x.size()[2] #目標是把圖片「放大」回原本的解析度，並且把剛剛流失的細節補回來
        diffX = x4.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1) #這是 U-Net 的關鍵「跳接 (Skip Connection)」，把下坡時保留的細節（x4）跟上坡的特徵（x）拼在一起，這樣 AI 畫出來的邊緣才會清晰。
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
        return torch.sigmoid(logits) #將輸出結果壓在 0 到 1 之間，變成「機率值」（例如：這個點有 90% 機率是神經）。

# ==========================================
# 2. 核心功能
# ==========================================
def calculate_dice(pred, target, smooth=1e-5): #用來計算 AI 考幾分。smooth=1e-5 是為了防止分母為 0 當掉
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0: return 1.0
    return (2. * intersection + smooth) / (union + smooth)

def overlay_mask(image_pil, mn, ft, ct):#這是畫圖大師。
    base = image_pil.convert("RGBA")
    
    # 顏色: 黃 / 藍 / 紅 (Alpha 150)
    color_mn = (255, 255, 0, 150) #定義顏色。MN(黃)、FT(藍)、CT(紅)，150 代表半透明度。
    color_ft = (0, 0, 255, 150)   
    color_ct = (255, 0, 0, 150)   

    def create_layer(mask, color):#是一個內部小函式，負責把單一通道的遮罩（Mask）變成有顏色的透明圖層。
        layer = Image.new("RGBA", base.size, color)
        # 修復過暗問題 (jpg fix)
        mask_uint8 = ((mask > 0.1) * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8).convert("L")
        # 尺寸對齊
        if mask_img.size != base.size:
            mask_img = mask_img.resize(base.size, Image.NEAREST)
        layer.putalpha(mask_img)
        return layer

    base = Image.alpha_composite(base, create_layer(ct, color_ct))#把 CT、FT、MN 一層一層蓋在原始圖片 (base) 上面。
    base = Image.alpha_composite(base, create_layer(ft, color_ft))
    base = Image.alpha_composite(base, create_layer(mn, color_mn))
    
    return base.convert("RGB")

# ==========================================
# 3. 滑桿版 GUI (Slider Mode)
# ==========================================
class SliderGUI:
    def __init__(self, root):#設定主視窗與聰明的路徑偵測，建立操控面板，建立螢幕牆(UI上的三個圖片螢幕)，並設定圖例(顏色)
        self.root = root
        self.root.title("Carpal Tunnel Player (Video Mode)")
        self.root.geometry("1400x850")
        
        # --- [新增] 路徑自動偵測邏輯 程式會變聰明，自動抓這支程式所在的資料夾 (base_dir)。
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')
        self.dataset_dir = os.path.join(self.base_dir, 'dataset')
        
        # 確保資料夾變數存在 (防呆)
        if not os.path.exists(self.checkpoints_dir):
            self.checkpoints_dir = self.base_dir
        if not os.path.exists(self.dataset_dir):
            self.dataset_dir = self.base_dir
        # -----------------------------
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data_list = [] 
        self.is_playing = False # 播放狀態
        
        # --- UI 左側控制區 --- #建立按鈕：Load Model、Select Folder、Play Animation
        self.left = tk.Frame(root, width=280, bg="#f0f0f0")
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(self.left, text="Step 1. 載入模型", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w", pady=5)
        tk.Button(self.left, text="Load Model", command=self.load_model_dialog).pack(fill=tk.X)
        self.lbl_model = tk.Label(self.left, text="Not Loaded", fg="red", bg="#f0f0f0")
        self.lbl_model.pack(anchor="w")

        tk.Label(self.left, text="Step 2. 選擇資料夾 (dataset/0)", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w", pady=(15,5))
        tk.Button(self.left, text="Select Folder", command=self.load_folder_dialog).pack(fill=tk.X)
        self.lbl_folder = tk.Label(self.left, text="No Folder", fg="blue", bg="#f0f0f0")
        self.lbl_folder.pack(anchor="w")

        # --- 播放控制區 ---
        tk.Label(self.left, text="Step 3. 影像控制", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w", pady=(20,5))
        
        # 顯示當前張數
        self.lbl_counter = tk.Label(self.left, text="Slice: 0 / 0", font=("Arial", 12), bg="#f0f0f0")
        self.lbl_counter.pack(pady=5)

        # 滑桿 (Scale)
        self.slider = tk.Scale(self.left, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_move, length=250)
        self.slider.pack(pady=5)

        # 播放按鈕
        self.btn_play = tk.Button(self.left, text="▶ Play Animation", command=self.toggle_play, bg="#dddddd", height=2)
        self.btn_play.pack(fill=tk.X, pady=10)

        # 序列分數
        self.frm_seq = tk.LabelFrame(self.left, text="Sequence DC (Mean)", bg="#f0f0f0")
        self.frm_seq.pack(fill=tk.X, pady=20)
        self.lbl_seq = tk.Label(self.frm_seq, text="-", bg="#f0f0f0")
        self.lbl_seq.pack()

        # --- UI 右側顯示區 ---
        self.right = tk.Frame(root)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.frm_imgs = tk.Frame(self.right)
        self.frm_imgs.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(self.frm_imgs, text="Original T1", font=("Arial", 12)).grid(row=0, column=0)
        tk.Label(self.frm_imgs, text="Ground Truth", font=("Arial", 12)).grid(row=0, column=1)
        tk.Label(self.frm_imgs, text="Prediction (AI)", font=("Arial", 12)).grid(row=0, column=2)

        self.sz = 380 # 放大一點顯示
        self.cv1 = tk.Canvas(self.frm_imgs, width=self.sz, height=self.sz, bg="black")
        self.cv1.grid(row=1, column=0, padx=5)
        self.cv2 = tk.Canvas(self.frm_imgs, width=self.sz, height=self.sz, bg="black")
        self.cv2.grid(row=1, column=1, padx=5)
        self.cv3 = tk.Canvas(self.frm_imgs, width=self.sz, height=self.sz, bg="black")
        self.cv3.grid(row=1, column=2, padx=5)

        self.lbl_cur_score = tk.Label(self.right, text="Current Slice DC: -", font=("Arial", 14, "bold"))
        self.lbl_cur_score.pack(pady=20)
        
        # Legend
        legend = tk.Frame(self.right)
        legend.pack()
        tk.Label(legend, text="■ MN (Yellow)", fg="#CCCC00", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(legend, text="■ FT (Blue)", fg="blue", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(legend, text="■ CT (Red)", fg="red", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        # [修改] 自動載入 (偵測 checkpoints 資料夾內的模型)
        default_model = os.path.join(self.checkpoints_dir, "best_model_fold_0.pth")
        if os.path.exists(default_model):
            self.load_model(default_model)
        else:
            # 備用：檢查當前目錄
            local_model = "best_model_fold_0.pth"
            if os.path.exists(local_model):
                self.load_model(local_model)

    def load_model_dialog(self):#打開檔案總管可以自己選資料夾(AI模型)
        # [修改] 預設打開 checkpoints 資料夾
        p = filedialog.askopenfilename(initialdir=self.checkpoints_dir, filetypes=[("Model", "*.pth")])
        if p: self.load_model(p)

    def load_model(self, p):#這個函式負責把選定的 .pth 檔案（模型的記憶與經驗）裝進 U-Net 網路架構裡
        try:
            self.model = UNet(n_channels=2, n_classes=3).to(self.device)
            self.model.load_state_dict(torch.load(p, map_location=self.device)) # 移除 weights_only 以相容舊版 torch
            self.model.eval()
            self.lbl_model.config(text=f"Loaded: {os.path.basename(p)}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def load_folder_dialog(self):#只能讀取資料夾 拿來選datatest的
        # [修改] 預設打開 dataset 資料夾
        f = filedialog.askdirectory(initialdir=self.dataset_dir)
        if f: self.load_data(f)

    def load_data(self, folder):
        self.data_list = []
        self.lbl_folder.config(text=os.path.basename(folder))
        
        t1_dir = os.path.join(folder, "T1")
        if not os.path.exists(t1_dir):
            messagebox.showerror("Error", "No T1 folder found!")
            return
            
        files = glob.glob(os.path.join(t1_dir, "*.jpg"))
        if not files: files = glob.glob(os.path.join(t1_dir, "*.png"))
        try: files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except: files.sort()

        if not files:
            messagebox.showwarning("Warning", "Empty folder!")
            return

        # === 自動判斷資料格式 ===
        # 檢查是否有 MN 資料夾 (舊格式)
        is_separate_mode = os.path.exists(os.path.join(folder, "MN"))
        
        if is_separate_mode:
            print(f"偵測到模式：[分開的資料夾] (MN/FT/CT)")
        else:
            print(f"偵測到模式：[單一 GT 資料夾] (GT)")

        # 建立清單
        for t1_path in files:
            fname = os.path.basename(t1_path)
            entry = {
                't1': t1_path,
                't2': os.path.join(folder, "T2", fname),
                'mode': 'separate' if is_separate_mode else 'combined' # 記住每一張圖的模式
            }
            
            if is_separate_mode:
                # 舊資料格式：分別記錄三個路徑
                entry['mn'] = os.path.join(folder, "MN", fname)
                entry['ft'] = os.path.join(folder, "FT", fname)
                entry['ct'] = os.path.join(folder, "CT", fname)
            else:
                # 新資料格式：記錄 GT 路徑
                entry['gt'] = os.path.join(folder, "GT", fname)
                
            self.data_list.append(entry)

        total = len(self.data_list)
        self.slider.config(to=total - 1)
        self.slider.set(0)
        self.lbl_counter.config(text=f"Slice: 1 / {total}")
        
        self.show_frame(0)
        self.root.after(100, self.calc_seq_dc)

    def get_tensor(self, idx):
        if idx >= len(self.data_list): return None, None
        d = self.data_list[idx]
        
        trans = transforms.ToTensor()
        try:
            t1 = Image.open(d['t1']).convert('L')
            t2 = Image.open(d['t2']).convert('L')
        except:
            return None, None

        if t1.size != t2.size: t2 = t2.resize(t1.size)
        inp = torch.cat((trans(t1), trans(t2)), dim=0).unsqueeze(0)
        
        # === 分流處理 ===
        if d['mode'] == 'separate':
            # --- 模式 A: 舊資料 (分開的資料夾) ---
            def read_mask(path):
                if os.path.exists(path):
                    img = Image.open(path).convert('L')
                    if img.size != t1.size: img = img.resize(t1.size)
                    t = trans(img)
                    return (t > 0.01).float()
                else:
                    return torch.zeros((1, t1.size[1], t1.size[0]))

            mn = read_mask(d['mn'])
            ft = read_mask(d['ft'])
            ct = read_mask(d['ct'])
            gt = torch.cat((mn, ft, ct), dim=0)
            
        else:
            # --- 模式 B: 新資料 (單一彩色 GT) ---
            gt_path = d['gt']
            if os.path.exists(gt_path):
                gt_color = Image.open(gt_path).convert('RGB')
                if gt_color.size != t1.size: 
                    gt_color = gt_color.resize(t1.size, Image.NEAREST)
                
                arr = np.array(gt_color)
                R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                
                # 1. 解析顏色 (洋紅=MN, 青=FT, 藍=CT)
                mn_mask_raw = (R > 100) & (B > 100) & (G < 100)
                ft_mask_raw = (G > 100) & (B > 100) & (R < 100)
                ct_mask_raw = (B > 100) & (R < 100) & (G < 100)

                # 2. 補洞修正：CT 包含 MN 和 FT
                ct_mask_fixed = ct_mask_raw | mn_mask_raw | ft_mask_raw

                # 3. 轉成 Tensor
                mn = torch.from_numpy(mn_mask_raw).float().unsqueeze(0)
                ft = torch.from_numpy(ft_mask_raw).float().unsqueeze(0)
                ct = torch.from_numpy(ct_mask_fixed).float().unsqueeze(0)
                
                gt = torch.cat((mn, ft, ct), dim=0)
            else:
                gt = torch.zeros((3, t1.size[1], t1.size[0]))

        return inp, gt

    def calc_seq_dc(self):#這個函式負責把目前載入的這位病人的每一張切片都跑過一次，計算 AI 預測得準不準，最後算出一個平均分。
        if not self.model or not self.data_list: return
        self.lbl_seq.config(text="Calculating...")
        self.root.update()
        
        scores = [[], [], []]
        with torch.no_grad():
            for i in range(len(self.data_list)):
                inp, gt = self.get_tensor(i)
                if inp is None: continue
                out = self.model(inp.to(self.device))
                out = (out > 0.5).float().cpu()
                for c in range(3):
                    scores[c].append(calculate_dice(out[0,c], gt[c]).item())

        m = [np.mean(s) if s else 0 for s in scores]
        self.lbl_seq.config(text=f"MN: {m[0]:.2f} | FT: {m[1]:.2f} | CT: {m[2]:.2f}")

    # --- 滑動事件 ---
    def on_slider_move(self, val):#回應使用者的滑鼠拖曳動作。
        idx = int(val)
        self.lbl_counter.config(text=f"Slice: {idx+1} / {len(self.data_list)}")
        self.show_frame(idx)

    # --- 顯示核心邏輯 ---
    def show_frame(self, idx): #取得數據與 AI 預測 (Data & Prediction)接著讓 AI 進行推論，並把結果轉回 CPU 準備畫圖
        if not self.model or not self.data_list: return
        
        inp, gt = self.get_tensor(idx)
        if inp is None: return
        
        # Predict
        with torch.no_grad():
            out = self.model(inp.to(self.device))
            out = (out > 0.5).float().cpu()
            
        # Score即時評分 這讓使用者可以觀察：是不是到了手腕某個特定深度的切面，AI 就會預測得比較差？
        s = [calculate_dice(out[0,i], gt[i]).item() for i in range(3)]
        self.lbl_cur_score.config(text=f"MN: {s[0]:.2f}  FT: {s[1]:.2f}  CT: {s[2]:.2f}")
        
        # Render
        t1_pil = transforms.ToPILImage()(inp[0,0])
        t1_pil = t1_pil.resize((self.sz, self.sz))
        
        self.ph1 = ImageTk.PhotoImage(t1_pil)
        self.cv1.create_image(0,0, anchor=tk.NW, image=self.ph1)
        
        gt_img = overlay_mask(t1_pil, gt[0].numpy(), gt[1].numpy(), gt[2].numpy())
        self.ph2 = ImageTk.PhotoImage(gt_img.resize((self.sz, self.sz)))
        self.cv2.create_image(0,0, anchor=tk.NW, image=self.ph2)
        
        pred_img = overlay_mask(t1_pil, out[0,0].numpy(), out[0,1].numpy(), out[0,2].numpy())
        self.ph3 = ImageTk.PhotoImage(pred_img.resize((self.sz, self.sz)))
        self.cv3.create_image(0,0, anchor=tk.NW, image=self.ph3)

    # --- 播放控制 ---
    def toggle_play(self): #播放/暫停的開關
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Play Animation")
        else:
            self.is_playing = True
            self.btn_play.config(text="⏸ Stop Animation")
            threading.Thread(target=self.play_loop, daemon=True).start()

    def play_loop(self):#自動播放引擎
        while self.is_playing:
            cur = self.slider.get()
            nxt = cur + 1
            if nxt > self.slider.cget("to"):
                nxt = 0 # Loop back to start
            
            self.slider.set(nxt) # 這會觸發 on_slider_move -> show_frame
            time.sleep(0.1) # 控制播放速度 (0.1秒一張)

if __name__ == "__main__":#
    root = tk.Tk()
    app = SliderGUI(root)
    root.mainloop()
