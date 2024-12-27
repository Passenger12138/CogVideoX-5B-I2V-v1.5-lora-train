import filecmp

def are_files_identical(file_path_1, file_path_2):
    """
    比较两个文件是否完全相同。
    
    参数:
        file_path_1 (str): 第一个文件的路径。
        file_path_2 (str): 第二个文件的路径。
        
    返回:
        bool: 如果两个文件完全相同则返回 True，否则返回 False。
    """
    return filecmp.cmp(file_path_1, file_path_2, shallow=False)

if __name__ == "__main__":
    # 替换为你的 mp4 文件路径
    file1 = '/maindata/data/shared/public/haobang.geng/code/video-generate/CogVideo/inference/output/debug.mp4'
    file2 = '/maindata/data/shared/public/haobang.geng/code/video-generate/CogVideo/inference/output/debug2.mp4'

    if are_files_identical(file1, file2):
        print("两个MP4文件完全一样。")
    else:
        print("两个MP4文件不相同。")