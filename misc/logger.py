import cv2
import numpy as np
import os
from typing import Optional, Tuple


class VideoLogger:
    """Class to handle video recording and visualization for robotic tasks."""
    
    def __init__(self, save_dir: str):
        """
        Initialize the video logger.
        
        Args:
            save_dir: Directory to save videos
        """
        self.save_dir = save_dir
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_name: Optional[str] = None
        self.new_video_name: Optional[str] = None
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def start_recording(self, test_set: str, task_id: str, object_name: str, seed: int):
        """
        Start recording a new video.
        
        Args:
            test_set: Name of the test set
            task_id: Task ID
            object_name: Name of the object
            seed: Random seed
        """
        object_name_with_underline = object_name.replace(' ', '_')
        self.video_name = f'{self.save_dir}/{test_set}_{task_id}_pick_up_{object_name_with_underline}_{seed}.mp4'
        self.new_video_name = self.video_name.replace('.mp4', '_x264.mp4')
        
        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            self.video_name, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            20, 
            (256 * 2, 256)
        )
    
    def log_frame(self, obs: dict, bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Log a single frame with optional bounding box visualization.
        
        Args:
            obs: Observation dictionary containing images
            bbox: Optional bounding box tuple (front_bbox, wrist_bbox)
        """
        if self.video_writer is None:
            raise RuntimeError("Video recording not started. Call start_recording() first.")
        
        # Get images and flip them
        front_img = obs["front_view_image"][::-1]
        wrist_img = obs["side_view_image"][::-1]
        
        # Draw bounding boxes if provided
        if bbox is not None:
            front_img = self._draw_bbox(front_img, bbox[0])
            wrist_img = self._draw_bbox(wrist_img, bbox[1])
        
        # Concatenate images and convert color format
        combined_img = np.concatenate([front_img, wrist_img], axis=1)
        combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        
        # Write frame
        self.video_writer.write(combined_img)
    
    def _draw_bbox(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Draw bounding box on image.
        
        Args:
            img: Input image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Image with bounding box drawn
        """
        # Resize bbox from 224 to 256
        resized_bbox = (bbox / 224 * 256).astype(int)
        img = img.astype(np.uint8)
        
        # Draw rectangle
        cv2.rectangle(
            img, 
            (resized_bbox[0], resized_bbox[1]), 
            (resized_bbox[2], resized_bbox[3]), 
            (0, 255, 0), 
            2
        )
        
        return img
    
    def stop_recording(self, success: bool = False):
        """
        Stop recording and finalize the video.
        
        Args:
            success: Whether the task was successful
        """
        if self.video_writer is None or self.new_video_name is None or self.video_name is None:
            return
        
        # Release video writer
        self.video_writer.release()
        self.video_writer = None
        
        # Update filename based on success
        if success:
            self.new_video_name = self.new_video_name.replace('.mp4', '_success.mp4')
        else:
            self.new_video_name = self.new_video_name.replace('.mp4', '_fail.mp4')
        
        # Convert video using ffmpeg
        self._convert_video()
        
        # Clean up original video file
        if os.path.exists(self.video_name):
            os.remove(self.video_name)
    
    def _convert_video(self):
        """Convert video using ffmpeg with x264 codec."""
        if self.video_name is None or self.new_video_name is None:
            return
        command = f'ffmpeg -i {self.video_name} -vcodec libx264 {self.new_video_name} -y > /dev/null 2>&1'
        os.system(command)
    
    def cleanup(self):
        """Clean up resources."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
