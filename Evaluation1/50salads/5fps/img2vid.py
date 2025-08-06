# imageから指定FPSで動画を生成する
# 画像の名前は連番 (ex: img_0000.jpg, img_0001.jpg, ...)
# ffmpegを使用する
import os
import subprocess

def create_video_from_images(image_folder, output_video, fps=30):
    # 画像のパスを取得
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    
    if not image_files:
        print("No images found in the specified folder.")
        return
    
    # 画像のフルパスを作成
    image_paths = [os.path.join(image_folder, img) for img in image_files]
    
    # ffmpegコマンドを作成
    command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(image_folder, 'img_%04d.jpg'),  # 連番の形式に合わせる
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    # ffmpegを実行
    subprocess.run(command)
    print(f"Video created: {output_video}")
    
    # 動画の作成が完了したら、動画のパスを返す
    return output_video

if __name__ == "__main__":
    list = ["01-2","02-1","02-2","03-1","03-2"]
    for i in list:
        image_folder = f'/home/yamaguchi/vmlserver06/Experiment/50salads/img/{i}'  # 画像フォルダのパスを指定
        video_name = f"rgb_{i}"
        output_video = f'./50salads/video/{video_name}.mp4'  # 出力動画のファイル名を指定
        fps = 5  # フレームレートを指定
        
        create_video_from_images(image_folder, output_video, fps)