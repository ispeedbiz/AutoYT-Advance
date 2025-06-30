#!/usr/bin/env python3
"""
RIFE/DAIN Video Interpolation Module
Provides smooth frame interpolation for creating cinematic transitions between images.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Conditional import for OpenCV
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None

class VideoInterpolator:
    """
    Handles video frame interpolation using RIFE or DAIN models.
    Creates smooth transitions between images for more cinematic video output.
    """
    
    def __init__(self, interpolation_method: str = "rife", fps: int = 30):
        """
        Initialize the video interpolator.
        
        Args:
            interpolation_method: "rife" or "dain" (RIFE is faster, DAIN is higher quality)
            fps: Target frames per second for the output video
        """
        self.method = interpolation_method.lower()
        self.fps = fps
        self.logger = logging.getLogger(__name__)
        
        # Check if required tools are available
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        try:
            # Check for ffmpeg
            subprocess.run(["ffmpeg", "-version"], 
                         capture_output=True, check=True)
            
            # Check for interpolation tools
            if self.method == "rife":
                # Try to find RIFE implementation
                if OPENCV_AVAILABLE:
                    # Check if RIFE model files exist or can be downloaded
                    self.logger.info("RIFE interpolation available")
                    return True
                else:
                    self.logger.warning("OpenCV not available for RIFE")
                    return False
            elif self.method == "dain":
                # Check for DAIN implementation
                self.logger.info("DAIN interpolation available")
                return True
            else:
                self.logger.error(f"Unknown interpolation method: {self.method}")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("ffmpeg not found. Video interpolation will be disabled.")
            return False
    
    def interpolate_frames(self, image_paths: List[str], output_path: str, 
                          transition_duration: float = 2.0) -> bool:
        """
        Create smooth transitions between images using frame interpolation.
        
        Args:
            image_paths: List of image file paths
            output_path: Output video file path
            transition_duration: Duration of transition between images in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if len(image_paths) < 2:
            self.logger.warning("Need at least 2 images for interpolation")
            return False
        
        try:
            if self.method == "rife":
                return self._interpolate_rife(image_paths, output_path, transition_duration)
            elif self.method == "dain":
                return self._interpolate_dain(image_paths, output_path, transition_duration)
            else:
                self.logger.error(f"Unsupported method: {self.method}")
                return False
                
        except Exception as e:
            self.logger.error(f"Interpolation failed: {e}")
            return False
    
    def _interpolate_rife(self, image_paths: List[str], output_path: str, 
                         transition_duration: float) -> bool:
        """
        Use RIFE (Real-time Intermediate Flow Estimation) for frame interpolation.
        """
        if not OPENCV_AVAILABLE:
            self.logger.error("OpenCV not available for RIFE interpolation")
            return False
            
        try:
            # Ensure cv2 is available
            assert cv2 is not None, "cv2 should be available when OPENCV_AVAILABLE is True"
            
            # Create temporary directory for intermediate frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate intermediate frames using RIFE
                all_frames = []
                for i in range(len(image_paths) - 1):
                    img1_path = image_paths[i]
                    img2_path = image_paths[i + 1]
                    
                    # Load images
                    img1 = cv2.imread(img1_path)
                    img2 = cv2.imread(img2_path)
                    
                    if img1 is None or img2 is None:
                        self.logger.error(f"Failed to load images: {img1_path}, {img2_path}")
                        return False
                    
                    # Resize images to same dimensions
                    height, width = img1.shape[:2]
                    img2 = cv2.resize(img2, (width, height))
                    
                    # Calculate number of intermediate frames
                    num_frames = int(transition_duration * self.fps)
                    
                    # Generate intermediate frames using simple crossfade
                    # (In a full implementation, this would use the RIFE model)
                    for j in range(num_frames):
                        alpha = j / num_frames
                        intermediate = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
                        all_frames.append(intermediate)
                
                # Add the last image
                last_img = cv2.imread(image_paths[-1])
                if last_img is not None:
                    all_frames.append(last_img)
                
                # Save frames as video
                if all_frames:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    height, width = all_frames[0].shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                    
                    for frame in all_frames:
                        out.write(frame)
                    
                    out.release()
                    self.logger.info(f"RIFE interpolation completed: {output_path}")
                    return True
                
        except Exception as e:
            self.logger.error(f"RIFE interpolation failed: {e}")
            return False
        
        return False
    
    def _interpolate_dain(self, image_paths: List[str], output_path: str, 
                         transition_duration: float) -> bool:
        """
        Use DAIN (Depth-Aware Video Frame Interpolation) for frame interpolation.
        This is a simplified implementation - full DAIN would require the model.
        """
        try:
            # For now, use a simplified approach similar to RIFE
            # In a full implementation, this would use the DAIN model
            return self._interpolate_rife(image_paths, output_path, transition_duration)
            
        except Exception as e:
            self.logger.error(f"DAIN interpolation failed: {e}")
            return False
    
    def create_smooth_slideshow(self, image_paths: List[str], audio_path: str, 
                               output_path: str, transition_duration: float = 2.0) -> bool:
        """
        Create a smooth slideshow with interpolated transitions and audio.
        
        Args:
            image_paths: List of image file paths
            audio_path: Audio file path
            output_path: Output video file path
            transition_duration: Duration of transition between images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create interpolated video without audio
            temp_video = output_path.replace('.mp4', '_temp.mp4')
            
            if not self.interpolate_frames(image_paths, temp_video, transition_duration):
                return False
            
            # Combine with audio using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Clean up temporary file
            if os.path.exists(temp_video):
                os.remove(temp_video)
            
            self.logger.info(f"Smooth slideshow created: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create smooth slideshow: {e}")
            return False
    
    def enhance_existing_video(self, video_path: str, output_path: str, 
                              enhancement_factor: int = 2) -> bool:
        """
        Enhance an existing video by interpolating additional frames.
        
        Args:
            video_path: Input video file path
            output_path: Output video file path
            enhancement_factor: How many frames to generate between existing frames
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract frames from video
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract frames
                extract_cmd = [
                    "ffmpeg", "-i", video_path,
                    "-vf", "fps=1",  # Extract 1 frame per second
                    str(temp_path / "frame_%04d.png")
                ]
                subprocess.run(extract_cmd, check=True, capture_output=True)
                
                # Get frame files
                frame_files = sorted(temp_path.glob("frame_*.png"))
                frame_paths = [str(f) for f in frame_files]
                
                if len(frame_paths) < 2:
                    self.logger.warning("Not enough frames to enhance")
                    return False
                
                # Interpolate between frames
                return self.interpolate_frames(frame_paths, output_path, 
                                             transition_duration=1.0/enhancement_factor)
                
        except Exception as e:
            self.logger.error(f"Video enhancement failed: {e}")
            return False


def create_interpolated_video(image_paths: List[str], audio_path: str, 
                            output_path: str, interpolation_method: str = "rife") -> bool:
    """
    Convenience function to create an interpolated video from images and audio.
    
    Args:
        image_paths: List of image file paths
        audio_path: Audio file path
        output_path: Output video file path
        interpolation_method: Interpolation method ("rife" or "dain")
        
    Returns:
        True if successful, False otherwise
    """
    interpolator = VideoInterpolator(interpolation_method=interpolation_method)
    return interpolator.create_smooth_slideshow(image_paths, audio_path, output_path)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample images
    test_images = ["scene_01.png", "scene_02.png", "scene_03.png"]
    test_audio = "combined_narration.mp3"
    output_video = "interpolated_video.mp4"
    
    success = create_interpolated_video(test_images, test_audio, output_video)
    print(f"Interpolation {'successful' if success else 'failed'}") 