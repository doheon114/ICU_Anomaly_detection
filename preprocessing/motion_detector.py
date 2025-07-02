import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

# ---------------------- GPU-based SSIM Function Definition ---------------------- #
def pytorch_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM (Structural Similarity Index Measure) between two images using PyTorch.
    
    Args:
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        window_size (int): Size of the window for calculating statistics
        size_average (bool): Whether to average the SSIM map
        
    Returns:
        float: SSIM score between 0 and 1
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map

def frame_similarity_gpu(frame1, frame2, device='cuda'):
    """
    Calculate similarity between two frames using GPU-accelerated SSIM.
    
    Args:
        frame1 (numpy.ndarray): First BGR frame
        frame2 (numpy.ndarray): Second BGR frame
        device (str): Device to run calculations on ('cuda' or 'cpu')
        
    Returns:
        float: Similarity score between 0 and 1
    """
    transform = transforms.ToTensor()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    tensor1 = transform(gray1).unsqueeze(0).to(device)
    tensor2 = transform(gray2).unsqueeze(0).to(device)

    return pytorch_ssim(tensor1, tensor2).item()

# ---------------------- Frame Extraction Functions ---------------------- #
def extract_unique_frames(video_path, output_dir, ssim_threshold=0.85, device='cuda'):
    """
    Extract unique frames from a video based on SSIM threshold.
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        ssim_threshold (float): Threshold for frame similarity (0-1)
        device (str): Device to run calculations on ('cuda' or 'cpu')
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)

    prev_frame = None
    frame_count = 0
    saved_seconds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        second = frame_count // frame_interval

        if frame_count % frame_interval == 0:
            if prev_frame is None:
                prev_frame = frame
                filename = f"second_{second:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                saved_seconds.append(second)
            else:
                similarity = frame_similarity_gpu(prev_frame, frame, device)
                if similarity < ssim_threshold:
                    filename = f"second_{second:05d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), frame)
                    saved_seconds.append(second)
                    prev_frame = frame

        frame_count += 1

    cap.release()
    print(f"Saved {len(saved_seconds)} frames to {output_dir}")

def extract_from_all_videos_recursive(input_root, output_root, ssim_threshold=0.85, device='cuda'):
    """
    Recursively process all .mp4 files in the input directory and its subdirectories.
    
    Args:
        input_root (str): Root directory containing video files
        output_root (str): Root directory for saving extracted frames
        ssim_threshold (float): Threshold for frame similarity (0-1)
        device (str): Device to run calculations on ('cuda' or 'cpu')
    """
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.mp4'):
                full_video_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_video_path, input_root)
                video_output_dir = os.path.join(output_root, os.path.splitext(rel_path)[0])
                print(f"Processing: {rel_path}")
                extract_unique_frames(full_video_path, video_output_dir, ssim_threshold, device)

# ---------------------- Usage Example ---------------------- #
if __name__ == "__main__":
    input_root = "/home/work/.doheon114/ICU_Anomaly/dataset/SICU2/rgb"
    output_root = "./extracted_frames"
    extract_from_all_videos_recursive(input_root, output_root, ssim_threshold=0.85, device='cuda')
