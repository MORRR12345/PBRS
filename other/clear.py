# 清理训练数据文件夹中的旧数据和无用数据

import re
import shutil
from datetime import datetime
from pathlib import Path

def clean_train_folders(
    base_dir="model",
    folder_pattern=r"Train_data_(\d{2})-(\d{2})-(\d{2})-(\d{2})",
    check_subfolder="picture",     # 只检查这个子文件夹中的 PDF
    keep_latest=10,
):
    """清理训练数据文件夹"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"目录不存在: {base_dir}")
        return

    # Step 1. 找出符合命名规则的文件夹
    target_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and re.fullmatch(folder_pattern, folder.name):
            try:
                m, d, h, mi = map(int, re.findall(r"\d+", folder.name))
                folder_time = datetime(datetime.now().year, m, d, h, mi)
                target_folders.append((folder, folder_time))
            except ValueError:
                continue

    if not target_folders:
        print("未找到符合命名规则的文件夹。")
        return

    # Step 2. 删除 picture 文件夹下无 PDF 的文件夹
    def has_pdf_files(path: Path):
        check_path = path / check_subfolder
        if not check_path.exists():
            return False
        return any(f.suffix.lower() == ".pdf" for f in check_path.rglob("*"))

    valid_folders = []
    for folder, t in target_folders:
        if has_pdf_files(folder):
            valid_folders.append((folder, t))
        else:
            print(f"🗑️ 删除无 PDF 文件的文件夹: {folder.name}")
            shutil.rmtree(folder, ignore_errors=True)

    # Step 3. 保留最新的 keep_latest 个文件夹
    valid_folders.sort(key=lambda x: x[1], reverse=True)
    for folder, t in valid_folders[keep_latest:]:
        print(f"🕒 删除旧文件夹 ({t.strftime('%m-%d %H:%M')}): {folder.name}")
        shutil.rmtree(folder, ignore_errors=True)

    print("✅ 清理完成。")

# =============================
if __name__ == "__main__":
    clean_train_folders(
        base_dir="model",
        folder_pattern=r"Train_data_(\d{2})-(\d{2})-(\d{2})-(\d{2})",
        check_subfolder="picture",
        keep_latest=10,
    )
