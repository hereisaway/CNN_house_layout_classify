import os


def delete_empty_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否存在且大小为0
        if os.path.isfile(file_path) and os.path.getsize(file_path) < 5000:
            # os.remove(file_path)  # 删除文件
            print(f"Deleted: {file_path}")


# 使用示例
folder_path = 'data/2'  # 替换为你的文件夹路径
delete_empty_files(folder_path)