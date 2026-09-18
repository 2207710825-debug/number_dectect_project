# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/17
@Describe ：合并两个文件夹中的图像数据到新的目标文件夹中
'''

import os
import shutil


def merge_folders(src_dir1, src_dir2, output_dir):
    """
    将两个源文件夹中的文件合并到一个全新的文件夹中

    :param src_dir1: 第一个源文件夹路径
    :param src_dir2: 第二个源文件夹路径
    :param output_dir: 新建的合并目标文件夹路径
    """
    # 1. 校验源文件夹是否存在
    if not os.path.exists(src_dir1):
        raise FileNotFoundError(f"源文件夹 1 不存在: {src_dir1}")
    if not os.path.exists(src_dir2):
        raise FileNotFoundError(f"源文件夹 2 不存在: {src_dir2}")

    # 2. 如果输出文件夹已存在，清空并重新创建；不存在则新建
    if os.path.exists(output_dir):
        print(f"[提示] 目标合并文件夹已存在，正在清理覆盖: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    # 记录已复制的文件名列表，防重名覆盖
    existing_files = set()
    total_copied = 0

    # 内部复制辅助函数
    def copy_files_from_source(src_path_dir, tag_prefix):
        nonlocal total_copied
        files = [f for f in os.listdir(src_path_dir) if f.lower().endswith(valid_extensions)]
        print(f"正在读取 [{src_path_dir}]，共找到 {len(files)} 个文件...")

        for fname in files:
            src_file = os.path.join(src_path_dir, fname)
            dst_file = os.path.join(output_dir, fname)

            # 处理同名文件冲突：如果文件名已存在，重命名为 '原名_tag.png'
            if fname in existing_files:
                base_name, ext = os.path.splitext(fname)
                new_fname = f"{base_name}_{tag_prefix}{ext}"
                dst_file = os.path.join(output_dir, new_fname)
                print(f"  └─ 冲突重命名: {fname} -> {new_fname}")
                existing_files.add(new_fname)
            else:
                existing_files.add(fname)

            shutil.copy(src_file, dst_file)
            total_copied += 1

    # 3. 依次合并两个文件夹
    print("\n=== 开始合并流程 ===")
    copy_files_from_source(src_dir1, tag_prefix="dir1")
    copy_files_from_source(src_dir2, tag_prefix="dir2")

    print("\n" + "=" * 50)
    print(f"文件夹合并完成！")
    print(f"合并后新文件夹路径: {output_dir}")
    print(f"新文件夹中的文件总数: {total_copied} 张")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 需要合并的两个源文件夹路径
    DIR1 = r"E:\pythondemo\number_dectect_project\imagedata\training_img"
    DIR2 = r"E:\pythondemo\number_dectect_project\imagedata\training_img_augmented"

    # 合并后生成的新文件夹路径
    MERGED_DIR = r"E:\pythondemo\number_dectect_project\imagedata\training_augmented"

    # 执行合并
    merge_folders(DIR1, DIR2, MERGED_DIR)