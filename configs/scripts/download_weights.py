import os
import gdown

FILES = [
    ("1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT", "stylegan2-ffhq-config-f.pt"),
    ("1N0MZSqPRJpLfP4mFQCS14ikrVSe8vQlL", "model_ir_se50.pth"),
]

def main(out_dir="weights"):
    os.makedirs(out_dir, exist_ok=True)

    for fid, fname in FILES:
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[skip] {fname}")
            continue

        url = f"https://drive.google.com/uc?id={fid}"
        print(f"[download] {fname}")
        gdown.download(url, out_path, quiet=False)

    print("[ok] weights:", os.listdir(out_dir))

if __name__ == "__main__":
    main()