import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sys
import os
import json
from datetime import datetime

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

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(config_path, checkpoint_path, device='cpu'):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    return Model().to(device)

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(
            config_path='configs/deim_rtdetrv2/deim_r50vd_60e_coco.yml',
            checkpoint_path='pretrained/deim_rtdetrv2_l.pth',
            device=DEVICE
        )
    return MODEL

def process_detections(labels, boxes, scores, threshold=0.4):
    scr = scores[0]
    lab = labels[0][scr > threshold]
    box = boxes[0][scr > threshold]
    scrs = scr[scr > threshold]
    
    detections = []
    for j, (b, l, s) in enumerate(zip(box, lab, scrs)):
        detection = {
            "class": COCO_CLASSES[l.item()],
            "confidence": float(s.item()),
            "bbox": [float(x) for x in b.tolist()]
        }
        detections.append(detection)
    
    return detections

def save_detection_results(frame_results, source_type, source_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{source_type}_{source_id}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(frame_results, f, indent=4)
    
    return filepath

def process_frame(frame):
    if frame is None:
        return None
    
    model = get_model()
    
    # Convert frame to PIL image
    if isinstance(frame, np.ndarray):
        if frame.ndim == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame_pil = Image.fromarray(frame)
    else:
        frame_pil = frame
        frame = np.array(frame)
    
    w, h = frame_pil.size
    orig_size = torch.tensor([[w, h]]).to(DEVICE)
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    im_data = transforms(frame_pil).unsqueeze(0).to(DEVICE)
    
    # Get model predictions
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)
    
    # Process detections
    detections = process_detections(labels, boxes, scores)
    
    # Draw predictions on frame
    draw = ImageDraw.Draw(frame_pil)
    for det in detections:
        bbox = det["bbox"]
        draw.rectangle(bbox, outline='red', width=2)
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.text((bbox[0], bbox[1]), text=label, fill='red')
    
    # Convert back to numpy array
    return np.array(frame_pil)

def process_video(video_path):
    if video_path is None:
        return None
    
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = os.path.join(RESULTS_DIR, os.path.basename(video_path).rsplit('.', 1)[0] + '_processed.mp4')
    
    # Use MJPG codec which is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    temp_output = os.path.join(RESULTS_DIR, 'temp_output.avi')
    
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(DEVICE)
            
            transforms = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            im_data = transforms(frame_pil).unsqueeze(0).to(DEVICE)
            
            # Get model predictions
            with torch.no_grad():
                labels, boxes, scores = model(im_data, orig_size)
            
            # Process detections
            detections = process_detections(labels, boxes, scores)
            
            # Draw predictions
            for det in detections:
                bbox = det["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            out.write(frame)
            frame_count += 1
            
    finally:
        cap.release()
        out.release()
        
        # Convert AVI to MP4 using ffmpeg
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-y',
                output_path
            ], check=True)
            os.remove(temp_output)  # Clean up temporary file
        except Exception as e:
            # If ffmpeg fails, just return the AVI file
            output_path = temp_output
    
    return output_path

def create_interface():
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # AI-Powered Object Detection
        
        This application provides real-time object detection using the DEIM model.
        """)
        
        with gr.Tab("Video Upload"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Processed Video")
            with gr.Row():
                process_btn = gr.Button("Process Video", variant="primary")
                status = gr.Textbox(label="Status", value="Ready")
            
            def process_video_with_status(video):
                try:
                    output = process_video(video)
                    return output, f"Processing complete! Results saved in {RESULTS_DIR}"
                except Exception as e:
                    return None, f"Error: {str(e)}"
            
            process_btn.click(
                fn=process_video_with_status,
                inputs=[video_input],
                outputs=[video_output, status]
            )
        
        with gr.Tab("Webcam Stream"):
            gr.Markdown("""
            ### Instructions:
            1. Click 'Start' to begin webcam capture
            2. The model will detect objects in real-time
            3. Detections are shown with bounding boxes and confidence scores
            """)
            
            with gr.Row():
                # Updated webcam component for Gradio 5.x
                camera = gr.Image(
                    type="numpy",
                    label="Camera Input",
                    sources=["webcam"],
                    interactive=True,
                    streaming=True
                )
                stream_output = gr.Image(
                    type="numpy",
                    label="Processed Stream"
                )
            
            camera.stream(
                fn=process_frame,
                inputs=[camera],
                outputs=[stream_output],
                show_progress=False
            )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 