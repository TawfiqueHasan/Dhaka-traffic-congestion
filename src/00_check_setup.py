import os, sys, urllib.request

def check_gpu():
    print("="*55); print("STEP 1 - Checking PyTorch + CUDA (your GPU)"); print("="*55)
    try:
        import torch
    except ImportError:
        print("PyTorch not installed."); sys.exit(1)
    print(f"PyTorch version : {torch.__version__}")
    ok = torch.cuda.is_available()
    print(f"CUDA available  : {ok}")
    if ok:
        print(f"GPU detected    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        print(f"VRAM            : {vram:.1f} GB")
        print("\nYour GPU is set up correctly.")
    else:
        print("\nCUDA NOT available - torch will use CPU (slow).")
    return ok

def run_yolo(ok):
    print("\n"+"="*55); print("STEP 2 - Loading pretrained YOLOv8"); print("="*55)
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("Pretrained YOLOv8n loaded.")
    print("\n"+"="*55); print("STEP 3 - Detection on a sample image"); print("="*55)
    os.makedirs("data", exist_ok=True)
    p = os.path.join("data","sample_test.jpg")
    if not os.path.exists(p):
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", p)
    device = 0 if ok else "cpu"
    r = model(p, device=device)[0]
    print(f"\nRan on device  : {'GPU (cuda:0)' if ok else 'CPU'}")
    print(f"Objects found  : {len(r.boxes)}")
    for b in r.boxes:
        print(f"   - {model.names[int(b.cls)]:12s} conf={float(b.conf):.2f}")
    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports","first_detection.jpg")
    r.save(filename=out)
    print(f"\nAnnotated image saved to: {out}")

if __name__ == "__main__":
    ok = check_gpu(); run_yolo(ok)
    print("\nPHASE 0 COMPLETE - environment verified.")
