import subprocess
import os, math, json
import multiprocessing


def process_video_split_by_length(src_path, dest_path, video_length_sec):
    """处理单个视频文件，如果视频长度超过video_length_sec，则将其切分为多个视频片段，每个片段提取帧到不同文件夹"""
    try:
        # 获取视频总时长
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                    '-of', 'default=noprint_wrappers=1:nokey=1', src_path]
        result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        # 计算需要分割的片段数
        num_segments = max(1, int(math.ceil(duration / video_length_sec)))
        
        for segment in range(num_segments):
            # 计算当前片段的起始时间和结束时间
            start_time = segment * video_length_sec
            end_time = (segment + 1) * video_length_sec
            segment_dir = dest_path + f"-{start_time}s_{end_time}s"
            if os.path.exists(segment_dir):
                print(f"已存在: {segment_dir}")
                continue
            os.makedirs(segment_dir, exist_ok=True)
            
            # 执行ffmpeg命令，以1fps提取帧，只提取当前时间段的帧
            cmd = [
                'ffmpeg', '-loglevel', 'error',
                '-i', src_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-threads','1',
                '-vf', 'fps=1',
                '-q:v', '5',  # 质量设置，5为中等质量
                os.path.join(segment_dir, "%06d.jpg")  # 使用六位数命名
            ]
            subprocess.run(cmd)
    except Exception as e:
        print(f"{src_path}, {e}")


if __name__ == '__main__':
    video_length_sec = 180
    src_base_folder = '/data/wangzhichao/datasets/EgoExoLearn/videos'
    dest_base_folder = '/data/wangzhichao/datasets/EgoExoLearn/frames_time'
    commands = list()

    for video_fname in os.listdir(src_base_folder):
        src_path = os.path.join(src_base_folder, video_fname)
        dest_path = os.path.join(dest_base_folder, '.'.join(video_fname.split('.')[:-1]))
        commands.append((src_path, dest_path, video_length_sec))

    print(f"{len(commands)=}")
    with multiprocessing.Pool(20) as p:
        p.starmap(process_video_split_by_length, commands)
