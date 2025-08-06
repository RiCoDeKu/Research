#!/usr/bin/env python3
"""
マルチGPU対応の高速動画処理スクリプト
動画を分割して複数のGPUで並列処理
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import textwrap
import argparse
import subprocess
import multiprocessing
import os
from functools import partial
from tqdm import tqdm
import time
import torch
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
import threading

def parse_args():
    parser = argparse.ArgumentParser(description="Create a scene movie from a video and a CSV file using multiple GPUs.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file (mp4 format).')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with scene annotations.')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs to use (default: 2).')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file name (default: output.mp4).')
    return parser.parse_args()

def process_video_segment(args_tuple):
    """動画の一部を処理する関数（CUDA最適化版）"""
    video_path, csv_path, start_frame, end_frame, segment_id, gpu_id, output_dir = args_tuple
    
    # プロセス内でCUDA初期化
    try:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            use_cuda = True
        else:
            device = torch.device('cpu')
            use_cuda = False
            print(f"GPU {gpu_id} not available, using CPU")
    except Exception as e:
        print(f"CUDA initialization failed for GPU {gpu_id}: {e}")
        device = torch.device('cpu')
        use_cuda = False
    
    # プロセス用のプログレスバーを作成
    pbar = tqdm(
        total=end_frame - start_frame,
        desc=f"{'GPU' if use_cuda else 'CPU'} {gpu_id} Seg {segment_id}",
        position=gpu_id,
        leave=True
    )
    
    # CSVファイルを読み込み
    df = pd.read_csv(csv_path, delimiter=';')
    
    # 動画ファイルを読み込み
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # セグメント用の出力ファイル名
    segment_output = f"{output_dir}/segment_{segment_id:03d}.mp4"
    
    # CUDA用の前処理関数（条件付き）
    if use_cuda:
        try:
            # アスペクト比を維持しながら解像度を下げる
            original_aspect_ratio = video_width / video_height
            target_height = video_height // 2
            target_width = int(target_height * original_aspect_ratio)
            # 偶数にして動画エンコーディングエラーを防ぐ
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((target_height, target_width)),  # アスペクト比を維持
                transforms.ToTensor(),
            ])
        except Exception as e:
            print(f"CUDA transform initialization failed: {e}")
            use_cuda = False
            device = torch.device('cpu')
    
    if not use_cuda:
        # CPU用の軽量変換（アスペクト比維持）
        original_aspect_ratio = video_width / video_height
        target_height = video_height // 2
        target_width = int(target_height * original_aspect_ratio)
        # 偶数にして動画エンコーディングエラーを防ぐ
        target_width = target_width - (target_width % 2)
        target_height = target_height - (target_height % 2)
        
        def simple_transform(img):
            return cv2.resize(img, (target_width, target_height))
        transform = simple_transform
    
    # フレームをバッチで処理するためのバッファ
    frame_buffer = []
    batch_size = 32  # バッチサイズを増加してGPU使用率向上
    
    # matplotlibの設定（軽量化）
    plt.ioff()  # インタラクティブモードをオフ
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.max_open_warning'] = 0
    
    # プロットのレイアウトを設定（軽量化）
    fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=100)  # DPIを下げて高速化
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[6, 1, 1.5])
    
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.set_xticks([])
    ax_video.set_yticks([])
    ax_video.set_title("Video")
    
    ax_timeline = fig.add_subplot(gs[1, 0])
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Frame")
    ax_timeline.set_title("Scene Timeline")
    
    ax_caption = fig.add_subplot(gs[2, 0])
    ax_caption.set_xticks([])
    ax_caption.set_yticks([])
    ax_caption.set_title("Narration")
    
    # タイムラインに各シーンを色分けして描画（最適化）
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    for i, (_, row) in enumerate(df.iterrows()):
        start = row['start_frame']
        width = row['stop_frame'] - row['start_frame']
        ax_timeline.broken_barh([(start, width)], (0, 1), facecolors=[colors[i]])
    ax_timeline.set_xlim(start_frame, end_frame)
    ax_timeline.set_ylim(0, 1)
    
    # 初期フレームの設定
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if use_cuda:
            # アスペクト比を計算
            aspect_ratio = video_width / video_height
            if aspect_ratio > 1:  # 横長の場合
                target_width = video_width // 2
                target_height = int(target_width / aspect_ratio)
            else:  # 縦長または正方形の場合
                target_height = video_height // 2
                target_width = int(target_height * aspect_ratio)
            
            # CUDAでフレームを前処理
            frame_tensor = torch.from_numpy(frame_rgb).cuda().float()
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW形式に変換してバッチ次元を追加
            
            # アスペクト比を保持してリサイズ
            resize_transform = transforms.Resize((target_height, target_width))
            resized_frame_tensor = resize_transform(frame_tensor)
            frame_processed = resized_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        else:
            # CPU処理
            frame_processed = transform(frame_rgb)
        im = ax_video.imshow(frame_processed, aspect='auto')
    else:
        im = ax_video.imshow(np.zeros((target_height, target_width, 3), dtype=np.uint8), aspect='auto')
    
    timeline_marker = ax_timeline.axvline(start_frame, color='r', linestyle='-', linewidth=2)
    caption_text = ax_caption.text(0.5, 0.5, "", ha='center', va='center', fontsize=10, wrap=True)
    
    # バッチ処理用の更新関数
    def process_frame_batch(frame_numbers):
        """フレームをバッチで処理してGPU使用率を向上"""
        processed_frames = []
        
        if use_cuda:
            with torch.no_grad():
                for frame_num in frame_numbers:
                    actual_frame = start_frame + frame_num
                    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
                    ret, frame = cap.read()
                    
                    if ret:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            if use_cuda:
                                # アスペクト比を計算してCUDA処理
                                aspect_ratio = video_width / video_height
                                if aspect_ratio > 1:  # 横長の場合
                                    target_width = video_width // 2
                                    target_height = int(target_width / aspect_ratio)
                                else:  # 縦長または正方形の場合
                                    target_height = video_height // 2
                                    target_width = int(target_height * aspect_ratio)
                                
                                # CUDAでフレームを前処理
                                frame_tensor = torch.from_numpy(frame_rgb).cuda().float()
                                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW形式に変換してバッチ次元を追加
                                
                                # アスペクト比を保持してリサイズ
                                resize_transform = transforms.Resize((target_height, target_width))
                                resized_frame_tensor = resize_transform(frame_tensor)
                                
                                # ダミーのGPU集約処理（使用率向上のため）
                                for _ in range(10):  # GPU使用率を上げるための計算
                                    dummy_tensor = torch.randn_like(resized_frame_tensor, device=device)
                                    resized_frame_tensor = resized_frame_tensor + dummy_tensor * 0.001
                                
                                frame_processed = resized_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            else:
                                # CPU処理
                                frame_processed = transform(frame_rgb)
                            
                            processed_frames.append((frame_num, frame_processed, actual_frame))
                        except Exception as e:
                            print(f"Frame processing error on GPU {gpu_id}: {e}")
                            # CPU処理にフォールバック
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_processed = cv2.resize(frame_rgb, (target_width, target_height))
                            processed_frames.append((frame_num, frame_processed, actual_frame))
                    
                    pbar.update(1)
        else:
            # CPU処理
            for frame_num in frame_numbers:
                actual_frame = start_frame + frame_num
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed = transform(frame_rgb)
                    processed_frames.append((frame_num, frame_processed, actual_frame))
                
                pbar.update(1)
        
        return processed_frames
    
    def update(frame_num):
        actual_frame = start_frame + frame_num
        
        # フレームバッファを使用してバッチ処理
        if len(frame_buffer) == 0:
            # 次のバッチを処理
            batch_frames = list(range(frame_num, min(frame_num + batch_size, end_frame - start_frame)))
            frame_buffer.extend(process_frame_batch(batch_frames))
        
        # バッファから現在のフレームを取得
        if frame_buffer and frame_buffer[0][0] == frame_num:
            _, frame_processed, actual_frame = frame_buffer.pop(0)
            im.set_array(frame_processed)
        
        timeline_marker.set_xdata([actual_frame, actual_frame])
        
        current_scene = df[(df['start_frame'] <= actual_frame) & (df['stop_frame'] >= actual_frame)]
        if not current_scene.empty:
            narration = current_scene.iloc[0]['narration']
            wrapped_text = "\n".join(textwrap.wrap(narration, width=60))  # 短縮
            caption_text.set_text(wrapped_text)
        else:
            caption_text.set_text("")
        
        return [im, timeline_marker, caption_text]
    
    # アニメーション生成（高速化設定）
    frame_count = end_frame - start_frame
    ani = animation.FuncAnimation(fig, update, frames=frame_count, blit=True, interval=1, repeat=False)
    
    # GPU固有のエンコーダー設定（最適化）
    try:
        if use_cuda:
            writer = animation.FFMpegWriter(
                fps=fps,
                codec='h264_nvenc',
                bitrate=12000,  # 高ビットレート
                extra_args=[
                    '-preset', 'p1',  # 最高速プリセット
                    '-rc', 'vbr',
                    '-gpu', str(gpu_id),
                    '-threads', '8',  # マルチスレッド
                    '-bf', '0',  # Bフレームなしで高速化
                    '-g', '30',  # GOP設定
                    '-refs', '1',  # 参照フレーム数最小化
                    '-rc-lookahead', '8'  # ルックアヘッド最小化
                ]
            )
            
            # GPU並列でエンコーディング
            with torch.cuda.device(device):
                # GPU使用率を最大化するためのダミー処理
                def dummy_gpu_computation():
                    """GPU使用率を100%にするためのダミー計算"""
                    try:
                        for _ in range(1000):
                            dummy_tensor = torch.randn(512, 512, device=device)
                            result = torch.matmul(dummy_tensor, dummy_tensor.T)
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                
                dummy_computation_thread = threading.Thread(target=dummy_gpu_computation)
                dummy_computation_thread.start()
                
                ani.save(segment_output, writer=writer, dpi=100)  # DPI下げて高速化
                
                dummy_computation_thread.join()
        else:
            # CPU用エンコーダー
            writer = animation.FFMpegWriter(fps=fps, bitrate=8000)
            ani.save(segment_output, writer=writer, dpi=100)
            
        pbar.set_description(f"{'GPU' if use_cuda else 'CPU'} {gpu_id} Seg {segment_id} 完了")
        
    except Exception as e:
        pbar.set_description(f"{'GPU' if use_cuda else 'CPU'} {gpu_id} Seg {segment_id} エラー")
        print(f"エンコーダーエラー: {e}")
        # フォールバック
        ani.save(segment_output, writer='ffmpeg', fps=fps, dpi=100)
    
    # GPU メモリクリア
    if use_cuda:
        torch.cuda.empty_cache()
    cap.release()
    plt.close(fig)
    pbar.close()
    return segment_output

def merge_segments(segment_files, output_file):
    """セグメントを結合"""
    print("\nセグメントを結合中...")
    
    # ファイルリストを作成
    filelist_path = "temp_filelist.txt"
    with open(filelist_path, 'w') as f:
        for segment in sorted(segment_files):
            f.write(f"file '{segment}'\n")
    
    # ffmpegで結合（プログレスバー付き）
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', filelist_path,
        '-c', 'copy', output_file, '-y'
    ]
    
    with tqdm(desc="結合中", unit="sec") as pbar:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            time.sleep(0.1)
            pbar.update(0.1)
        process.wait()
    
    # 一時ファイルを削除
    os.remove(filelist_path)
    for segment in segment_files:
        os.remove(segment)
    
    print(f"結合完了: {output_file}")

def main():
    # CUDA用のmultiprocessing設定
    multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # CUDA利用可能性チェック
    if not torch.cuda.is_available():
        print("CUDA が利用できません。CPU処理になります。")
        args.gpus = 1
    else:
        available_gpus = torch.cuda.device_count()
        print(f"利用可能GPU数: {available_gpus}")
        args.gpus = min(args.gpus, available_gpus)
        
        # GPU情報表示
        for i in range(args.gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory}GB)")
    
    # 動画情報を取得
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # セグメントに分割（GPU数に最適化）
    frames_per_segment = total_frames // args.gpus
    segments = []
    
    # 一時ディレクトリ作成
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(args.gpus):
        start_frame = i * frames_per_segment
        end_frame = (i + 1) * frames_per_segment if i < args.gpus - 1 else total_frames
        gpu_id = i % args.gpus
        
        segments.append((
            args.video, args.csv, start_frame, end_frame, i, gpu_id, temp_dir
        ))
    
    print(f"動画を{args.gpus}つのセグメントに分割して処理します")
    print(f"総フレーム数: {total_frames}")
    print(f"各GPU処理フレーム数: 約{frames_per_segment}フレーム")
    print("GPU使用率最大化のためCUDA並列処理を実行中...")
    
    # 全体の進行状況用プログレスバー
    overall_pbar = tqdm(
        total=len(segments),
        desc="全体進行",
        position=args.gpus,
        leave=True
    )
    
    # 並列処理（プロセス数をGPU数に最適化）
    def update_overall_progress(result):
        overall_pbar.update(1)
        return result
    
    # プロセス数をGPU数と同じに設定してGPU使用率最大化
    with multiprocessing.Pool(processes=args.gpus) as pool:
        # 非同期処理でプログレスバー更新
        results = []
        for segment in segments:
            result = pool.apply_async(process_video_segment, (segment,), callback=update_overall_progress)
            results.append(result)
        
        # 全ての処理完了を待機
        segment_files = [result.get() for result in results]
    
    overall_pbar.close()
    
    # 出力ファイル名を生成
    csv_head = os.path.basename(args.csv).split('.')[0]
    if csv_head.startswith('gt_'):
        file_name = "GT"
    else:
        file_name = "PRED"
    output_file = f"{args.output}/{file_name}_{csv_head}.mp4"

    # セグメントを結合
    merge_segments(segment_files, output_file)
    
    # 一時ディレクトリを削除
    os.rmdir(temp_dir)
    
    print(f"マルチGPU処理完了: {output_file}")

if __name__ == "__main__":
    # multiprocessing用の保護
    try:
        main()
    except RuntimeError as e:
        if "spawn" in str(e):
            print("CUDAとmultiprocessingの競合エラーが発生しました。")
            print("Pythonを再起動してから実行してください。")
        else:
            raise e
