import time
import torch
import torch.nn as nn
from PIL import Image
import mss
from torchvision import transforms
from collections import deque, Counter
import pickle
import pygetwindow as gw

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class DigitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class_names = ['0', '1', '10', '11', '12', '13', '2', '3', '4', '5', '6', '7', '8', '9', '-']


def scale_regions(regions, base_res, target_res):
    scale_x = target_res[0] / base_res[0]
    scale_y = target_res[1] / base_res[1]
    return [(int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)) for (x1, y1, x2, y2) in
            regions]


def capture(region):
    x1, y1, x2, y2 = region
    monitor = {'top': y1, 'left': x1, 'width': x2 - x1, 'height': y2 - y1}
    with mss.mss() as sct:
        img = sct.grab(monitor)
    return Image.frombytes('RGB', img.size, img.rgb)


def init():
    global regions, config_data, num_classes, run_model_path
    global device, model, target_FPS, target_dt, detected_history

    regions_1440p = [(1068, 44, 1128, 84), (1433, 44, 1493, 84)]
    base_res = (2560, 1440)

    windows = gw.getWindowsWithTitle('VALORANT')
    if windows:
        win = windows[0]
        target_res = (win.width, win.height)
        regions = scale_regions(regions_1440p, base_res, target_res)
    else:
        regions = []

    with open("config.pkl", "rb") as f:
        config_data = pickle.load(f)

    num_classes = config_data.get("num_classes")
    run_model_path = config_data.get("run_model_path")

    target_FPS = config_data.get("target_FPS", 30)
    target_dt = 1 / target_FPS
    detected_history = deque(maxlen=config_data.get("max_detected_history", 20))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN(num_classes).to(device)
    model.load_state_dict(torch.load(run_model_path, map_location=device))
    model.eval()

    return regions


def detect():
    while True:
        start = time.perf_counter()

        imgs = [transform(capture(r)).unsqueeze(0) for r in regions]
        batch = torch.cat(imgs).to(device)

        with torch.no_grad():
            preds = model(batch).argmax(dim=1).tolist()

        detected_history.append(preds)

        per_region = list(zip(*detected_history))
        avg_detected = []
        for region in per_region:
            most_common = Counter(region).most_common(1)
            val = most_common[0][0] if most_common else 0
            if val >= len(class_names) or val < 0:
                val = 0
            avg_detected.append(val)

        elapsed = time.perf_counter() - start
        time.sleep(max(0, target_dt - elapsed))

        total_time = time.perf_counter() - start
        FPS = 1 / total_time if total_time > 0 else 0

        preds_safe = [i if 0 <= i < len(class_names) else 0 for i in preds]

        print(
            f"Detected labels: {[class_names[i] for i in preds_safe]} | ≈ {[class_names[i] for i in avg_detected]} | FPS: {FPS:.2f}")
