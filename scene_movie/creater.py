import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import textwrap
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create a scene movie from a video and a CSV file.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file (mp4 format).')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with scene annotations.')
    return parser.parse_args()

args = parse_args()
# --- 設定項目 ---
# 動画ファイルのパスを指定
VIDEO_FILE_PATH = args.video
# CSVファイルのパスを指定
CSV_FILE_PATH = args.csv
# 出力する動画のファイル名
CSV_FILE_HEAD = CSV_FILE_PATH.split('/')[-1].split('.')[0]  # CSVファイル名から拡張子を除去
if CSV_FILE_HEAD.startswith('gt_'):
    FILE_NAME = "GT"
else:
    FILE_NAME = "PRED"
OUTPUT_VIDEO_FILE = f"{FILE_NAME}_{CSV_FILE_HEAD}.mp4"
# --- 設定項目はここまで ---

# CSVファイルを読み込み
try:
    df = pd.read_csv(CSV_FILE_PATH, delimiter=';')
except FileNotFoundError:
    print(f"エラー: CSVファイルが見つかりません: {CSV_FILE_PATH}")
    exit()

# 動画ファイルを読み込み
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    print(f"エラー: 動画ファイルを開けません: {VIDEO_FILE_PATH}")
    exit()

# 動画のプロパティを取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("動画情報を読み込みました:")
print(f"  - 合計フレーム数: {total_frames}")
print(f"  - FPS: {fps:.2f}")
print(f"  - 解像度: {video_width}x{video_height}")


# Matplotlibのフォントを設定 (日本語が文字化けする場合)
# 使用可能な日本語フォントにご自身の環境に合わせて変更してください。
# 例: 'IPAexGothic', 'Yu Gothic', 'Meiryo' など
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'


# プロットのレイアウトを設定
fig = plt.figure(figsize=(10, 8), constrained_layout=True)
# グリッドの高さの比率を調整 (映像 : タイムライン : キャプション)
gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[6, 1, 1.5])

# 1. 映像表示用サブプロット
ax_video = fig.add_subplot(gs[0, 0])
ax_video.set_xticks([])
ax_video.set_yticks([])
ax_video.set_title("Video")

# 2. タイムライン表示用サブプロット
ax_timeline = fig.add_subplot(gs[1, 0])
ax_timeline.set_yticks([])
ax_timeline.set_xlabel("Frame")
ax_timeline.set_title("Scene Timeline")

# 3. キャプション表示用サブプロット
ax_caption = fig.add_subplot(gs[2, 0])
ax_caption.set_xticks([])
ax_caption.set_yticks([])
ax_caption.set_title("Narration")


# タイムラインに各シーンを色分けして描画
colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
for i, (_, row) in enumerate(df.iterrows()):
    start_frame = row['start_frame']
    width = row['stop_frame'] - row['start_frame']
    ax_timeline.broken_barh([(start_frame, width)], (0, 1), facecolors=[colors[i]])
ax_timeline.set_xlim(0, total_frames)
ax_timeline.set_ylim(0, 1)

# アニメーションの初期化
# 空の画像、タイムラインの線、キャプションのテキストを準備
ret, frame = cap.read()
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = ax_video.imshow(frame_rgb, aspect='auto')
else:
    # 最初のフレームが読めない場合はダミーデータで初期化
    im = ax_video.imshow(np.zeros((video_height, video_width, 3), dtype=np.uint8), aspect='auto')

timeline_marker = ax_timeline.axvline(0, color='r', linestyle='-', linewidth=2)
caption_text = ax_caption.text(0.5, 0.5, "", ha='center', va='center', fontsize=12, wrap=True)


# アニメーション更新用関数
def update(frame_num):
    # 1. 映像の更新
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im.set_array(frame_rgb)
    
    # 2. タイムラインマーカーの更新
    timeline_marker.set_xdata([frame_num, frame_num])
    
    # 3. キャプションの更新
    current_scene = df[(df['start_frame'] <= frame_num) & (df['stop_frame'] >= frame_num)]
    if not current_scene.empty:
        narration = current_scene.iloc[0]['narration']
        # テキストが長すぎる場合に折り返す
        wrapped_text = "\n".join(textwrap.wrap(narration, width=80))
        caption_text.set_text(wrapped_text)
    else:
        caption_text.set_text("")
        
    # 進捗状況を表示
    if frame_num % 100 == 0:
        print(f"Processing frame {frame_num}/{total_frames}...")

    return [im, timeline_marker, caption_text]

# アニメーションの生成
# total_framesを小さくすると、テスト用に短い動画を生成できます
# 例: frames=300
ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

# アニメーションの保存
print("\nアニメーションをMP4ファイルとして保存しています。これには時間がかかる場合があります...")
try:
    # GPU情報を確認して、マルチGPU対応のNVIDIAエンコーダーを使用する
    import subprocess
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode('utf-8').strip().split('\n')
    print(f"検出されたGPU数: {len(gpu_info)}")
    for i, gpu in enumerate(gpu_info):
        print(f"  GPU {i}: {gpu}")
    
    if not gpu_info or gpu_info == ['']:
        raise RuntimeError("NVIDIA GPUが検出されません。通常のFFmpegエンコーダーを使用します。")
    
    # マルチGPU対応のエンコーダー設定
    # GPU 0を使用してメインエンコーディング、GPU数に応じてパフォーマンス設定を調整
    if len(gpu_info) >= 2:
        print("マルチGPU環境を検出しました。最適化された設定を使用します。")
        extra_args = [
            '-preset', 'p1',  # 最高速プリセット
            '-rc', 'vbr',
            '-cq', '23',      # 高品質設定
            '-b:v', '10M',    # 高ビットレート
            '-maxrate', '15M',
            '-bufsize', '20M',
            '-gpu', '0'       # メインGPUを指定
        ]
    else:
        extra_args = ['-preset', 'fast', '-rc', 'vbr']
    
    # NVIDIA GPU エンコーダーを使用して高速処理
    writer = animation.FFMpegWriter(
        fps=fps,
        codec='h264_nvenc',
        bitrate=8000,  # マルチGPU環境では高ビットレートを使用
        extra_args=extra_args
    )
    ani.save(OUTPUT_VIDEO_FILE, writer=writer, dpi=150)
    print(f"\n保存が完了しました: {OUTPUT_VIDEO_FILE}")
except Exception as e:
    print(f"\nh264_nvencでの保存に失敗しました: {e}")
    print("通常のエンコーダーで再試行します...")
    try:
        # フォールバック: 通常のFFmpegエンコーダー
        ani.save(OUTPUT_VIDEO_FILE, writer='ffmpeg', fps=fps, dpi=150)
        print(f"\n保存が完了しました: {OUTPUT_VIDEO_FILE}")
    except Exception as e2:
        print("\nエラー: アニメーションの保存に失敗しました。")
        print("FFmpegがシステムにインストールされ、PATHが通っていることを確認してください。")
        print(f"詳細: {e2}")

# リソースの解放
cap.release()