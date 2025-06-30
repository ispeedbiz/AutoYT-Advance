import requests
import json
import time
import os
from pathlib import Path

API_URL = "http://127.0.0.1:8188"
ULTRA_WORKFLOW_FILE = "workflow/prompt_ultra_quality.json"

def generate_ultra_quality_image(prompt_text, out_file="ultra_quality_result.png", negative_prompt=None, max_wait_sec=1200):
    """
    Generate ultra-high quality images optimized for slow-motion video applications.
    
    Features:
    - SDXL Base model with optimized settings
    - 1536x864 base resolution
    - 50 steps with DPM++ 2M
    - 1080p upscaling (1920x1080)
    - Optimized for cinematic quality
    """
    
    with open(ULTRA_WORKFLOW_FILE, "r") as f:
        workflow = json.load(f)

    # Update prompts
    positive_found = False
    for node in workflow.values():
        if node.get("class_type") == "CLIPTextEncode":
            if not positive_found and node["inputs"].get("text", "").strip():
                node["inputs"]["text"] = prompt_text
                positive_found = True
            elif negative_prompt and positive_found:
                node["inputs"]["text"] = negative_prompt
                break

    print("🚀 Sending ultra-quality prompt to ComfyUI...")
    print(f"📝 Prompt: {prompt_text[:100]}...")
    print(f"🎯 Target: 1080p resolution for slow-motion video")
    
    res = requests.post(f"{API_URL}/prompt", json={"prompt": workflow})
    res.raise_for_status()
    prompt_id = res.json()["prompt_id"]
    
    print(f"🔄 Ultra-quality generation in progress (prompt_id: {prompt_id})...")
    print("⏱️ Expected time: 2-3 minutes for high-quality generation")

    output_filename = None
    waited = 0
    check_interval = 5  # Check every 5 seconds
    
    while waited < max_wait_sec:
        time.sleep(check_interval)
        waited += check_interval
        
        try:
            hist_response = requests.get(f"{API_URL}/history/{prompt_id}")
            if hist_response.status_code != 200:
                continue
                
            hist = hist_response.json()
            
            if prompt_id in hist:
                prompt_history = hist[prompt_id]
                
                if "outputs" in prompt_history:
                    for node_id, node_output in prompt_history["outputs"].items():
                        if "images" in node_output and node_output["images"]:
                            output_filename = node_output["images"][0]["filename"]
                            break
                    
                    if output_filename:
                        print(f"✅ Ultra-quality image generated: {output_filename}")
                        break
                        
                elif "status" in prompt_history and prompt_history["status"].get("completed", False):
                    print("⚠️ Generation completed but no images found")
                    break
                    
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            continue
            
        # Show progress
        if waited % 30 == 0:  # Every 30 seconds
            print(f"🔄 Still generating ultra-quality image... ({waited}s/{max_wait_sec}s)")

    if not output_filename:
        print(f"❌ Ultra-quality generation timed out after {waited}s")
        return False

    # Download the generated image
    try:
        print(f"📥 Downloading ultra-quality image: {output_filename}")
        img_response = requests.get(f"{API_URL}/view?filename={output_filename}&type=output")
        
        if img_response.status_code == 200:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            with open(out_file, "wb") as f:
                f.write(img_response.content)
            
            if os.path.exists(out_file) and os.path.getsize(out_file) > 2000:  # At least 2KB
                file_size = os.path.getsize(out_file)
                print(f"✅ Ultra-quality image saved: {out_file}")
                print(f"📊 File size: {file_size // 1024}KB")
                print(f"🎬 Ready for slow-motion video processing")
                return True
            else:
                print(f"❌ Ultra-quality image file is too small or corrupted")
                return False
        else:
            print(f"❌ Failed to download ultra-quality image: HTTP {img_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading ultra-quality image: {e}")
        return False

def generate_slow_motion_sequence(prompt_base, num_frames=8, output_folder="slow_motion_sequence"):
    """
    Generate a sequence of ultra-quality images for slow-motion video.
    Each frame has slight variations to create smooth motion.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    
    print(f"🎬 Generating slow-motion sequence: {num_frames} frames")
    print(f"📁 Output folder: {output_folder}")
    
    for i in range(num_frames):
        # Add slight variations for motion
        motion_variations = [
            "slight movement, dynamic pose",
            "moment captured, natural motion",
            "flowing movement, organic motion",
            "gentle motion, subtle movement",
            "smooth transition, fluid motion",
            "natural flow, dynamic moment",
            "organic movement, life-like motion",
            "cinematic motion, film-like movement"
        ]
        
        frame_prompt = f"{prompt_base}, {motion_variations[i % len(motion_variations)]}"
        output_file = os.path.join(output_folder, f"frame_{i+1:03d}.png")
        
        print(f"🎨 Generating frame {i+1}/{num_frames}...")
        success = generate_ultra_quality_image(
            prompt_text=frame_prompt,
            out_file=output_file,
            max_wait_sec=1200
        )
        
        if success:
            image_paths.append(output_file)
            print(f"✅ Frame {i+1} completed")
        else:
            print(f"❌ Frame {i+1} failed")
    
    print(f"\n🎬 Slow-motion sequence complete!")
    print(f"✅ Generated {len(image_paths)}/{num_frames} frames")
    print(f"📁 Ready for video processing in: {output_folder}")
    
    return image_paths

def create_slow_motion_video(image_folder, audio_path, output_video="slow_motion_video.mp4", fps=24):
    """
    Create a slow-motion video from the generated image sequence.
    """
    try:
        from moviepy import ImageSequenceClip, AudioFileClip
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        image_paths = [os.path.join(image_folder, f) for f in image_files]
        
        if not image_paths:
            print("❌ No images found for video creation")
            return False
        
        print(f"🎬 Creating slow-motion video from {len(image_paths)} frames...")
        
        # Create video clip
        clip = ImageSequenceClip(image_paths, fps=fps)
        
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            audio = AudioFileClip(audio_path)
            clip = clip.set_audio(audio)
            print(f"🎵 Audio added: {audio_path}")
        
        # Write video file
        clip.write_videofile(
            output_video,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k"
        )
        
        print(f"✅ Slow-motion video created: {output_video}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating slow-motion video: {e}")
        return False

if __name__ == "__main__":
    # Test ultra-quality generation
    test_prompt = "cinematic realism, elderly Indian farmer Kishan with white beard, golden hour lighting, professional photography, shallow depth of field, film grain, no text, no watermark"
    
    print("🧪 Testing ultra-quality image generation...")
    success = generate_ultra_quality_image(
        prompt_text=test_prompt,
        out_file="test_ultra_quality.png"
    )
    
    if success:
        print("✅ Ultra-quality generation test successful!")
    else:
        print("❌ Ultra-quality generation test failed") 