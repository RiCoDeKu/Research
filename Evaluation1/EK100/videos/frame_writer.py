#フレーム数を動画の右上に表示するためのコード
import cv2
import os

def write_frame_number(video_path, output_path):
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 動画のプロパティを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 出力用の動画ライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレーム番号を動画の右上に表示
        cv2.putText(frame, f'Frame: {frame_number}/{frame_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # フレームを書き込む
        out.write(frame)
        
        frame_number += 1

    # リソースを解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Add frame numbers to a video.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video with frame numbers.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.video):
        print(f"Error: The video file {args.video} does not exist.")
    else:
        write_frame_number(args.video, args.output)
        print(f"Frame numbers added to the video and saved to {args.output}.")