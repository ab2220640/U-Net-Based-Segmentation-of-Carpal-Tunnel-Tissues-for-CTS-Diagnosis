import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import glob
import time
import threading

# ==========================================
# 1. 模型定義 (U-Net) - 保持不變
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
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
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
# 2. 核心功能
# ==========================================
def calculate_dice(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0: return 1.0
    return (2. * intersection + smooth) / (union + smooth)

def overlay_mask(image_pil, mn, ft, ct):
    base = image_pil.convert("RGBA")
    
    # 顏色: 黃 / 藍 / 紅 (Alpha 150)
    color_mn = (255, 255, 0, 150) 
    color_ft = (0, 0, 255, 150)   
    color_ct = (255, 0, 0, 150)   

    def create_layer(mask, color):
        layer = Image.new("RGBA", base.size, color)
        # 修復過暗問題 (jpg fix)
        mask_uint8 = ((mask > 0.1) * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8).convert("L")
        # 尺寸對齊
        if mask_img.size != base.size:
            mask_img = mask_img.resize(base.size, Image.NEAREST)
        layer.putalpha(mask_img)
        return layer

    base = Image.alpha_composite(base, create_layer(ct, color_ct))
    base = Image.alpha_composite(base, create_layer(ft, color_ft))
    base = Image.alpha_composite(base, create_layer(mn, color_mn))
    
    return base.convert("RGB")

# ==========================================
# 3. 滑桿版 GUI (Slider Mode)
# ==========================================
class SliderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Carpal Tunnel Player (Video Mode)")
        self.root.geometry("1400x850")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data_list = [] 
        self.is_playing = False # 播放狀態
        
        # --- UI 左側控制區 ---
        self.left = tk.Frame(root, width=280, bg="#f0f0f0")
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(self.left, text="Step 1. 載入模型", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w", pady=5)
        tk.Button(self.left, text="Load Model", command=self.load_model_dialog).pack(fill=tk.X)
        self.lbl_model = tk.Label(self.left, text="Not Loaded", fg="red", bg="#f0f0f0")
        self.lbl_model.pack(anchor="w")

        tk.Label(self.left, text="Step 2. 選擇資料夾 (data/0)", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w", pady=(15,5))
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

        # 自動載入
        if os.path.exists("best_model_fold_0.pth"):
            self.load_model("best_model_fold_0.pth")

    def load_model_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.pth")])
        if p: self.load_model(p)

    def load_model(self, p):
        try:
            self.model = UNet(n_channels=2, n_classes=3).to(self.device)
            self.model.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            self.model.eval()
            self.lbl_model.config(text=f"Loaded: {os.path.basename(p)}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_folder_dialog(self):
        f = filedialog.askdirectory()
        if f: self.load_data(f)

    def load_data(self, folder):
        self.data_list = []
        self.lbl_folder.config(text=os.path.basename(folder))
        
        t1_dir = os.path.join(folder, "T1")
        if not os.path.exists(t1_dir):
            messagebox.showerror("Error", "No T1 folder!")
            return
            
        files = glob.glob(os.path.join(t1_dir, "*.jpg"))
        
        # 【關鍵】排序：確保 0, 1, 2... 10 順序正確 (依照檔名數字)
        try:
            files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except:
            files.sort() # 如果檔名不是純數字，就用一般排序

        if not files:
            messagebox.showwarning("Warning", "Empty folder!")
            return

        # 建立清單
        for t1_path in files:
            fname = os.path.basename(t1_path)
            entry = {
                't1': t1_path,
                't2': os.path.join(folder, "T2", fname),
                'mn': os.path.join(folder, "MN", fname),
                'ft': os.path.join(folder, "FT", fname),
                'ct': os.path.join(folder, "CT", fname)
            }
            self.data_list.append(entry)

        # 設定滑桿範圍
        total = len(self.data_list)
        self.slider.config(to=total - 1)
        self.slider.set(0) # 回到第一張
        self.lbl_counter.config(text=f"Slice: 1 / {total}")
        
        # 載入完後先顯示第一張
        self.show_frame(0)
        
        # 計算總分
        self.root.after(100, self.calc_seq_dc)

    def get_tensor(self, idx):
        if idx >= len(self.data_list): return None, None
        d = self.data_list[idx]
        
        trans = transforms.ToTensor()
        t1 = Image.open(d['t1']).convert('L')
        t2 = Image.open(d['t2']).convert('L')
        if t1.size != t2.size: t2 = t2.resize(t1.size)
        inp = torch.cat((trans(t1), trans(t2)), dim=0).unsqueeze(0)
        
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
        return inp, gt

    def calc_seq_dc(self):
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
    def on_slider_move(self, val):
        idx = int(val)
        self.lbl_counter.config(text=f"Slice: {idx+1} / {len(self.data_list)}")
        self.show_frame(idx)

    # --- 顯示核心邏輯 ---
    def show_frame(self, idx):
        if not self.model or not self.data_list: return
        
        inp, gt = self.get_tensor(idx)
        if inp is None: return
        
        # Predict
        with torch.no_grad():
            out = self.model(inp.to(self.device))
            out = (out > 0.5).float().cpu()
            
        # Score
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
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Play Animation")
        else:
            self.is_playing = True
            self.btn_play.config(text="⏸ Stop Animation")
            threading.Thread(target=self.play_loop, daemon=True).start()

    def play_loop(self):
        while self.is_playing:
            cur = self.slider.get()
            nxt = cur + 1
            if nxt > self.slider.cget("to"):
                nxt = 0 # Loop back to start
            
            self.slider.set(nxt) # 這會觸發 on_slider_move -> show_frame
            time.sleep(0.1) # 控制播放速度 (0.1秒一張)

if __name__ == "__main__":
    root = tk.Tk()
    app = SliderGUI(root)
    root.mainloop()