import gradio as gr
import torch
import cv2
import numpy as np
import tempfile
import os
import requests
from urllib.parse import urlparse
import sys
import torchvision.transforms as T
from PIL import Image

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from engine.core import YAMLConfig

# Global model instance and device
MODEL = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_model():
    global MODEL
    if MODEL is None:
        config_path = os.path.join(os.path.dirname(__file__), '../configs/rtdetrv2_l.yaml')
        checkpoint_path = os.path.join(os.path.dirname(__file__), '../pretrained/deim_rtdetrv2_l.pth')
        
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        cfg.model.load_state_dict(state)
        MODEL = cfg.model.to(DEVICE)
        MODEL.eval()

def download_video(url):
    """Download video from URL to temporary file"""
    try:
        temp_dir = tempfile.mkdtemp()
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = 'video.mp4'
        
        temp_path = os.path.join(temp_dir, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return temp_path
    except Exception as e:
        raise gr.Error(f"Failed to download video: {str(e)}")

def process_frame(frame):
    """Process a single frame with DEIM model"""
    if frame is None:
        return None
        
    # Convert frame to PIL Image
    frame_pil = Image.fromarray(frame)
    
    # Transform image
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare input
    img = transform(frame_pil).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = MODEL(img)
    
    # Process outputs and draw boxes
    frame = np.array(frame_pil)
    for output in outputs:
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)
                class_name = COCO_CLASSES[label - 1]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label_text = f'{class_name} {score:.2f}'
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_video(video_input, progress=gr.Progress()):
    """Process video file or URL"""
    try:
        # Load model if not loaded
        load_model()
        
        # If input is URL, download it first
        if isinstance(video_input, str) and (video_input.startswith('http://') or video_input.startswith('https://')):
            video_path = download_video(video_input)
        else:
            video_path = video_input
            
        # Create a temporary file for the output video
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'output.mp4')
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        for frame_idx in progress.tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = process_frame(frame)
            if processed_frame is not None:
                out.write(processed_frame)
        
        # Release everything
        cap.release()
        out.release()
        
        return output_path
    except Exception as e:
        raise gr.Error(f"Error processing video: {str(e)}")

def live_detection(frame):
    """Process live webcam feed"""
    if frame is None:
        return None
        
    # Load model if not loaded
    load_model()
    
    # Process frame
    return process_frame(frame)

# Create Gradio interface with tabs
with gr.Blocks(title="DEIM Object Detection") as iface:
    gr.Markdown("# DEIM Object Detection")
    
    with gr.Tabs():
        # Video Upload/URL Tab
        with gr.Tab("Video Processing"):
            gr.Markdown("Upload a video file or provide a video URL for object detection")
            with gr.Row():
                video_input = gr.Video(label="Input Video")
                video_url = gr.Textbox(label="Or enter video URL", placeholder="https://example.com/video.mp4")
            
            process_btn = gr.Button("Process Video")
            video_output = gr.Video(label="Processed Video")
            
            process_btn.click(
                fn=process_video,
                inputs=[video_input],
                outputs=video_output
            )
            
            video_url.submit(
                fn=process_video,
                inputs=[video_url],
                outputs=video_output
            )
        
        # Live Detection Tab
        with gr.Tab("Live Detection"):
            gr.Markdown("Use your webcam for real-time object detection")
            with gr.Row():
                live_input = gr.Image(sources=["webcam"], streaming=True)
                live_output = gr.Image()
            
            live_input.stream(
                fn=live_detection,
                inputs=live_input,
                outputs=live_output,
                show_progress=False
            )

if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0", server_port=3000) 