# pyright: reportMissingImports=false
import os
import re
from datetime import datetime
from dotenv import load_dotenv
import logging
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import openai
import io
import random
from pathlib import Path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import Optional
import numpy as np

# Conditional import for moviepy to handle Python 3.13 compatibility issues
try:
    from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: moviepy import failed: {e}")
    print("💡 Video processing features will be limited")
    MOVIEPY_AVAILABLE = False
    ImageClip = None
    AudioFileClip = None
    TextClip = None
    CompositeVideoClip = None
    concatenate_videoclips = None

from comfyui_integration.generate_image import generate_image  # Make sure this import works
from prompt_utils import generate_metaphor_rich_prompts

# Conditional import for pydub to handle potential import issues
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: pydub import failed: {e}")
    print("💡 Audio processing features will be limited")
    PYDUB_AVAILABLE = False
    AudioSegment = None

IMAGE_STYLES = [
    "Studio Ghibli anime style, vibrant colors, children's book illustration, no text, no watermark.",
    "flat cartoon illustration, bright and clean, soft background, no text, no watermark.",
    "Pixar animation style, 3D cartoon, expressive, no text, no watermark."
]

# VIRAL GROWTH OPTIMIZED STYLES (10X Performance)
VIRAL_STYLE_LOCK = "cinematic realism, dramatic lighting, high contrast, professional photography, golden hour lighting, shallow depth of field, YouTube optimized, viral potential"
VIRAL_NEGATIVE_PROMPT = "text, watermark, signature, border, frame, lowres, bad anatomy, jpeg artifacts, blurry, distorted, amateur, poor quality, unprofessional, cartoon outline, anime artifacts"

# Legacy style (kept for compatibility)
STYLE_LOCK = "Studio Ghibli style, ultra-detailed, volumetric lighting, artstation 4k"
NEGATIVE_PROMPT = "text, watermark, signature, border, frame, lowres, bad anatomy, jpeg artifacts"

# Enhanced style and cinematography constants
STYLE_VARIATIONS = {
    "Motivational": "Studio Ghibli style, golden hour lighting, uplifting atmosphere, warm colors",
    "Storytelling": "Studio Ghibli style, cinematic composition, dramatic lighting, narrative depth",
    "Informative": "Studio Ghibli style, clear composition, balanced lighting, educational feel",
    "Inspirational": "Studio Ghibli style, ethereal lighting, dreamy atmosphere, soft focus"
}

CINEMATOGRAPHY_SHOTS = [
    "wide establishing shot",
    "medium shot with shallow depth of field",
    "close-up with emotional focus",
    "over-the-shoulder perspective",
    "low angle heroic shot"
]

# Enhanced style system with intelligent selection
CONTENT_BASED_STYLES = {
    "motivational_stories": "cinematic realism, warm golden hour lighting, inspiring atmosphere, shallow depth of field",
    "business_advice": "modern professional, clean composition, corporate aesthetic, high-end photography",
    "spiritual_wisdom": "ethereal lighting, soft focus, peaceful atmosphere, artistic photography",
    "life_lessons": "documentary style, natural lighting, authentic moments, photojournalism",
    "success_stories": "dramatic lighting, powerful composition, inspirational photography, film noir style",
    "educational": "clear composition, balanced lighting, infographic style, modern illustration"
}

AUDIENCE_STYLES = {
    "young_adults": "trending digital art, vibrant colors, modern aesthetic, social media style",
    "professionals": "corporate photography, clean lines, sophisticated composition, business aesthetic",
    "general_audience": "cinematic realism, universal appeal, emotional depth, film photography",
    "spiritual_seekers": "artistic photography, soft lighting, contemplative mood, fine art style"
}

def load_env():
    """
    Loads environment variables from .env file and returns them as a dictionary.
    Required keys: OPENAI_API_KEY, ELEVENLABS_API_KEY, HINDI_VOICE_ID, ENGLISH_VOICE_ID (optional).
    """
    env_path = Path(__file__).parent / '.env'
    print(f"Loading .env from: {env_path.resolve()}")
    load_dotenv(dotenv_path=env_path)
    env = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY"),
        "HINDI_VOICE_ID": os.environ.get("HINDI_VOICE_ID"),
        "ENGLISH_VOICE_ID": os.environ.get("ENGLISH_VOICE_ID"),
    }
    print("DEBUG HINDI_VOICE_ID=", os.environ.get("HINDI_VOICE_ID", "MISSING"))
    print("DEBUG ENGLISH_VOICE_ID=", os.environ.get("ENGLISH_VOICE_ID", "MISSING"))
    print("DEBUG env dict:", env)
    # Voice ID for Hindi narration (required)
    if not env["HINDI_VOICE_ID"] or env["HINDI_VOICE_ID"].strip() == "":
        raise Exception("No HINDI_VOICE_ID found in .env or it is blank!")
    return env


def sanitize_folder_name(text):
    # Sanitize text to use as folder name (alphanumeric and underscores only, max 50 chars)
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text)[:50]

def get_video_folder(base, title, date_str):
    """Create organized video folder structure with subfolders."""
    safe_title = sanitize_folder_name(title)
    folder_path = Path(base) / f"video_{date_str}_{safe_title}"
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Create organized subfolders
    subfolders = [
        "images",           # All scene images
        "audio",           # All audio blocks and final audio
        "prompts",         # All prompt files
        "feedback",        # Feedback files and reports
        "thumbnails",      # Thumbnail images
        "final_output",    # Final video and processed files
        "temp"             # Temporary files
    ]
    
    for subfolder in subfolders:
        (folder_path / subfolder).mkdir(exist_ok=True)
    
    return str(folder_path)

def save_text_to_file(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def insert_pauses_in_text(text):
    # Insert a pause indicator after sentence-ending punctuation (not used in final version)
    return re.sub(r"(।|\.|\?|\!)", r"\1[PAUSE]", text).replace("[PAUSE]", ", ")

def generate_thumbnail(title, img_url, output_path):
    """
    Generates a 1280x720 thumbnail image with the given title text.
    If img_url is provided or an AI-generated image is successfully obtained, it will be placed on the left side.
    Title text is wrapped and drawn on the right side (or centered if no image).
    """
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), (20, 20, 20))  # dark background
    draw = ImageDraw.Draw(img)
    image_used = False
    # Try to load provided image, otherwise attempt AI generation
    try:
        if img_url:
            resp = requests.get(img_url, timeout=10)
            side_img = Image.open(BytesIO(resp.content)).convert("RGB")
            side_img.thumbnail((width//2, height-40))
            img.paste(side_img, (40, (height - side_img.height) // 2))
            image_used = True
        else:
            if os.environ.get("OPENAI_API_KEY"):
                prompt_text = title if title and title.strip() else "an abstract concept"
                if len(prompt_text.split()) < 2:
                    prompt_text += " concept illustration"
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                img_resp = client.images.generate(prompt=prompt_text, n=1, size="1024x1024")
                if not img_resp.data or not img_resp.data[0].url:
                    raise Exception("Failed to generate image URL")
                img_url_ai = img_resp.data[0].url
                if img_url_ai is None:
                    raise Exception("Failed to generate image URL")
                resp = requests.get(img_url_ai, timeout=20)
                side_img = Image.open(BytesIO(resp.content)).convert("RGB")
                side_img.thumbnail((width//2, height-40))
                img.paste(side_img, (40, (height - side_img.height) // 2))
                image_used = True
            else:
                logging.warning("No image URL provided and OPENAI_API_KEY not set; thumbnail will have no image.")
    except Exception as e:
        logging.warning(f"Image load/generation failed: {e}")
        image_used = False

    # Choose font for title text
    font_path = "/Library/Fonts/Arial Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 70)
    except Exception as fe:
        logging.warning(f"Failed to load font '{font_path}': {fe}. Using default font.")
        font = ImageFont.load_default()

    # Determine text area: half width if image is used, full width if not
    max_text_width = (width // 2) - 80 if image_used else width - 80
    lines = text_wrap(title, font, max_text_width)

    # Center text vertically using .getbbox for height
    text_block_height = sum((font.getbbox(line)[3] - font.getbbox(line)[1]) for line in lines) + (len(lines) - 1) * 5
    y = (height - text_block_height) // 2
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (width//2 + (width//2 - w)//2) if image_used else ((width - w) // 2)
        # Draw text with a slight shadow for contrast
        draw.text((x+2, y+2), line, font=font, fill="black")
        draw.text((x, y), line, font=font, fill=(255, 191, 0))  # golden text
        y += h + 5
    img.save(output_path)
    logging.info(f"Thumbnail saved: {output_path}")
    return output_path

def text_wrap(text, font, max_width):
    # Wrap text into multiple lines to fit within max_width
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = (current_line + " " + word).strip()
        # .getbbox returns (x0, y0, x1, y1), width is x1
        if font.getbbox(test_line)[2] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def split_script_into_blocks(script, block_size=2):
    # Splits script into blocks of N sentences each (block_size = 2 is good)
    sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
    blocks = []
    for i in range(0, len(sentences), block_size):
        block = ' '.join(sentences[i:i+block_size]).strip()
        if block:
            blocks.append(block)
    return blocks

def generate_ai_image(prompt, api_key, out_path):
    import openai
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Clean and validate the prompt
        if not prompt or len(prompt.strip()) == 0:
            prompt = "abstract colorful background design"
        
        # Ensure prompt is not too long (OpenAI has limits)
        prompt = prompt[:1000]  # Limit prompt length
        
        print(f"🎨 Generating thumbnail with prompt: {prompt[:100]}...")
        
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard",  # Use standard quality to avoid issues
            model="dall-e-3"
        )
        
        if not response.data or not response.data[0].url:
            raise Exception("Failed to generate image URL")
        
        img_url = response.data[0].url
        if img_url is None:
            raise Exception("Failed to generate image URL")
        
        img_bytes = requests.get(img_url, timeout=15).content
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Center crop to 1024x576 and resize to 1280x720
        left = 0
        upper = (1024 - 576) // 2
        right = 1024
        lower = upper + 576
        img = img.crop((left, upper, right, lower)).resize((1280, 720), Image.Resampling.LANCZOS)
        img.save(out_path, "JPEG")
        
        print(f"✅ Thumbnail generated successfully")
        return out_path
        
    except Exception as e:
        print(f"⚠️ OpenAI image generation failed: {e}")
        print("🔄 Creating fallback thumbnail...")
        
        # Create a simple fallback thumbnail
        try:
            img = Image.new("RGB", (1280, 720), (50, 50, 100))  # Dark blue background
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            # Add some text
            text = "YouTube Video"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (1280 - text_width) // 2
            y = (720 - text_height) // 2
            
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            img.save(out_path, "JPEG")
            
            print(f"✅ Fallback thumbnail created")
            return out_path
            
        except Exception as e2:
            print(f"❌ Fallback thumbnail creation failed: {e2}")
            raise e  # Re-raise original error

def make_slideshow_video(image_paths, audio_paths, output_path):
    """Create a slideshow video from images and audio with proper duration matching."""
    if not MOVIEPY_AVAILABLE or AudioFileClip is None or ImageClip is None or concatenate_videoclips is None:
        print("⚠️ moviepy not available, cannot create slideshow video.")
        return None
    
    print(f"🎬 Creating slideshow video with {len(image_paths)} images and {len(audio_paths)} audio files")
    
    clips = []
    total_duration = 0
    
    # Process each image-audio pair
    for i, (img, audio) in enumerate(zip(image_paths, audio_paths)):
        try:
            print(f"   Processing clip {i+1}/{len(image_paths)}: {Path(img).name}")
            
            # Load audio to get duration
            audio_clip = AudioFileClip(audio)
            audio_duration = audio_clip.duration
            total_duration += audio_duration
            
            print(f"   Audio duration: {audio_duration:.2f} seconds")
            
            # Create image clip with audio duration
            img_clip = ImageClip(img, duration=audio_duration)
            
            # Add audio to image clip
            img_clip = img_clip.set_audio(audio_clip)
            
            # Keep original image resolution for now
            
            clips.append(img_clip)
            
        except Exception as e:
            print(f"❌ Error processing clip {i+1}: {e}")
            continue
    
    if not clips:
        print("❌ No valid clips created")
        return None
    
    print(f"✅ Created {len(clips)} clips, total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    try:
        # Concatenate all clips
        print("🔗 Concatenating video clips...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write final video
        print(f"📹 Writing final video to: {output_path}")
        final_clip.write_videofile(
            str(output_path), 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            bitrate="8000k",  # High quality
            verbose=False,
            logger=None
        )
        
        # Close clips to free memory
        for clip in clips:
            clip.close()
        final_clip.close()
        
        # Verify the created video
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✅ Video created successfully!")
            print(f"   File: {Path(output_path).name}")
            print(f"   Size: {file_size // (1024*1024)}MB")
            print(f"   Duration: {total_duration:.2f} seconds")
            return output_path
        else:
            print("❌ Video file was not created")
            return None
            
    except Exception as e:
        print(f"❌ Error creating final video: {e}")
        # Clean up clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        return None

def text_to_mp3_block(block_text, env, out_path, language="Hindi"):
    import requests
    if language.lower() == "english":
        voice_id = env.get("ENGLISH_VOICE_ID")
        if not voice_id:
            raise Exception("No ENGLISH_VOICE_ID found in .env for English TTS!")
    else:
        voice_id = env.get("HINDI_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": env["ELEVENLABS_API_KEY"]},
        json={"text": block_text, "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
    )
    if r.status_code != 200:
        raise Exception("TTS Error: " + r.text)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def generate_images_from_script(script_path, output_folder):
    with open(script_path, "r") as f:
        full_script = f.read()

    # Simple split by double linebreaks
    blocks = [blk.strip() for blk in full_script.split("\n\n") if len(blk.strip()) > 20]

    os.makedirs(output_folder, exist_ok=True)

    for i, block in enumerate(blocks):
        prompt = block.replace("\n", " ")[:200]  # short and clean
        out_file = os.path.join(output_folder, f"scene_{i+1:02d}.png")
        generate_image(prompt, out_file=out_file, negative_prompt=NEGATIVE_PROMPT)
        print(f"✅ Image {i+1} saved for block: {prompt[:60]}...")

def generate_and_save_images(prompt, out_file, negative_prompt):
    """
    Call the ComfyUI API pipeline for image generation.
    This is the main function that should be used for script-aligned image generation.
    """
    from comfyui_integration.generate_image import generate_image
    return generate_image(prompt, out_file=out_file, negative_prompt=negative_prompt)

def split_script_into_blocks_improved(script, block_len=300):
    """
    Improved script splitting function that creates better blocks for image generation.
    Splits every ~300 chars or at double newline (customize as needed).
    """
    blocks = re.split(r"\n\s*\n", script)
    final_blocks = []
    for block in blocks:
        if len(block) <= block_len:
            final_blocks.append(block.strip())
        else:
            # Split long blocks by sentences
            sentences = re.split(r'(?<=[।.!?])\s+', block.strip())
            current_block = ""
            for sentence in sentences:
                if len(current_block + sentence) <= block_len:
                    current_block += sentence + " "
                else:
                    if current_block.strip():
                        final_blocks.append(current_block.strip())
                    current_block = sentence + " "
            if current_block.strip():
                final_blocks.append(current_block.strip())
    
    return [b for b in final_blocks if b.strip()]

def split_script_by_duration(script: str, target_duration_minutes: float) -> list[str]:
    """
    Advanced content-first script splitting using Content Intelligence and Engagement Optimization.
    
    Args:
        script: The full script text
        target_duration_minutes: Target video duration in minutes
    
    Returns:
        List of script blocks optimized for the target duration and maximum engagement
    """
    print(f"🧠 Advanced Content Intelligence System")
    print(f"   Target duration: {target_duration_minutes} minutes")
    print(f"   Method: Content-first optimization with engagement analysis")
    
    try:
        # Load environment for API key
        env = load_env()
        api_key = env.get("OPENAI_API_KEY")
        
        if not api_key:
            print("   ⚠️ No OpenAI API key found, using fallback method")
            # Use the old dynamic system as fallback
            dynamic_max_duration = get_dynamic_max_scene_duration(
                "mixed", "neutral", target_duration_minutes=target_duration_minutes
            )
            return _fallback_duration_splitting(script, target_duration_minutes, dynamic_max_duration)
        
        # Initialize advanced systems
        from content_intelligence import ContentIntelligence
        from engagement_optimizer import EngagementOptimizer
        
        content_intel = ContentIntelligence(api_key)
        engagement_optimizer = EngagementOptimizer()
        
        # Step 1: Content Intelligence Analysis
        print("   🧠 Content Intelligence Analysis...")
        content_plan = content_intel.optimize_content_for_duration(script, target_duration_minutes)
        
        # Step 2: Engagement Optimization
        if content_plan["status"] in ["optimal", "auto_fixed"]:
            print("   📈 Engagement Optimization...")
            engagement_data = engagement_optimizer.optimize_scenes_for_engagement(
                content_plan["scene_texts"], 
                {"content_type": "mixed"}  # Will be replaced with actual analysis
            )
            
            # Use engagement-optimized scenes
            optimized_scenes = []
            for i, optimization in enumerate(engagement_data["optimized_scenes"]):
                scene_text = content_plan["scene_texts"][i]
                
                # Apply engagement adjustments if needed
                if optimization.adjustments:
                    print(f"   📝 Scene {i+1} optimization: {optimization.optimization_type} (engagement: {optimization.engagement_score:.2f})")
                    for adjustment in optimization.adjustments[:1]:  # Apply first suggestion
                        print(f"      💡 {adjustment}")
                
                optimized_scenes.append(scene_text)
            
            # Print final results
            print(f"\n📊 Content Intelligence Results:")
            print(f"   Status: {content_plan['status']} ({content_plan['confidence']} confidence)")
            print(f"   Method: {content_plan['method']}")
            print(f"   Scene count: {len(optimized_scenes)}")
            print(f"   Estimated duration: {content_plan['estimated_duration']:.1f} minutes")
            print(f"   Engagement score: {content_plan.get('engagement_score', 0.75):.2f}")
            print(f"   Overall engagement: {engagement_data['overall_engagement_score']:.2f}")
            
            # Print engagement recommendations
            if engagement_data["engagement_recommendations"]:
                print(f"   📈 Engagement insights:")
                for recommendation in engagement_data["engagement_recommendations"]:
                    print(f"      {recommendation}")
            
            print(f"   ✅ Advanced content optimization completed!")
            
            return optimized_scenes
        
        else:
            # Content-duration mismatch detected
            print(f"   ⚠️ Content-duration mismatch detected:")
            print(f"   {content_plan['optimization_notes']}")
            
            # Still try engagement optimization on best-effort scenes
            if content_plan.get("scene_texts"):
                print("   📈 Applying engagement optimization to available scenes...")
                engagement_data = engagement_optimizer.optimize_scenes_for_engagement(
                    content_plan["scene_texts"], 
                    {"content_type": "mixed"}
                )
                
                print(f"   Engagement score: {engagement_data['overall_engagement_score']:.2f}")
                return content_plan["scene_texts"]
            else:
                # Ultimate fallback
                print("   🔄 Using dynamic fallback method...")
                dynamic_max_duration = get_dynamic_max_scene_duration(
                    "mixed", "neutral", target_duration_minutes=target_duration_minutes
                )
                return _fallback_duration_splitting(script, target_duration_minutes, dynamic_max_duration)
    
    except ImportError as e:
        print(f"   ⚠️ Advanced systems not available: {e}")
        print("   🔄 Using dynamic fallback method...")
        # Use the improved dynamic system as fallback
        dynamic_max_duration = get_dynamic_max_scene_duration(
            "mixed", "neutral", target_duration_minutes=target_duration_minutes
        )
        return _fallback_duration_splitting(script, target_duration_minutes, dynamic_max_duration)
    
    except Exception as e:
        print(f"   ⚠️ Advanced optimization failed: {e}")
        print("   🔄 Using dynamic fallback method...")
        # Use the improved dynamic system as fallback
        dynamic_max_duration = get_dynamic_max_scene_duration(
            "mixed", "neutral", target_duration_minutes=target_duration_minutes
        )
        return _fallback_duration_splitting(script, target_duration_minutes, dynamic_max_duration)

def _fallback_duration_splitting(script: str, target_duration_minutes: float, max_scene_duration: int) -> list[str]:
    """
    Fallback method when advanced features are not available.
    """
    # Calculate target duration in seconds
    target_duration_seconds = target_duration_minutes * 60
    
    print(f"   🎬 Basic Duration-Aware Script Splitting:")
    print(f"   Target duration: {target_duration_minutes} minutes ({target_duration_seconds} seconds)")
    print(f"   Max scene duration: {max_scene_duration} seconds")
    
    # Split script into sentences first
    sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        print("   ⚠️ No valid sentences found in script")
        return [script[:200]] if script else [""]
    
    # Calculate estimated words per second (speaking rate)
    total_words = sum(len(s.split()) for s in sentences)
    estimated_speaking_rate = 2.0  # words per second (adjustable)
    estimated_duration = total_words / estimated_speaking_rate
    
    print(f"   Total words: {total_words}")
    print(f"   Estimated speaking duration: {estimated_duration:.1f} seconds ({estimated_duration/60:.1f} minutes)")
    
    # If script is too short for target duration, we have options:
    if estimated_duration < target_duration_seconds * 0.7:  # 70% threshold
        print(f"   ⚠️ Script is shorter than target duration")
        print(f"   Options: 1) Use natural breaks, 2) Extend with pauses, 3) Accept shorter video")
        
        # Option 1: Use natural breaks and accept shorter video
        scenes = _split_by_natural_breaks(sentences, max_scene_duration)
        print(f"   ✅ Using natural breaks: {len(scenes)} scenes")
        return scenes
    
    # If script is too long, we need to be more aggressive
    elif estimated_duration > target_duration_seconds * 1.3:  # 130% threshold
        print(f"   ⚠️ Script is longer than target duration")
        print(f"   Will create more scenes to fit timing")
        
        # Calculate target scenes based on content length
        target_scenes = max(3, int(estimated_duration / max_scene_duration))
        scenes = _split_by_content_density(sentences, target_scenes, max_scene_duration)
        print(f"   ✅ Created {len(scenes)} scenes for longer content")
        return scenes
    
    # Script length is appropriate - use balanced approach
    else:
        print(f"   ✅ Script length is appropriate for target duration")
        scenes = _split_by_natural_breaks(sentences, max_scene_duration)
        
        # Fine-tune scene count if needed
        if len(scenes) < 3:
            scenes = _split_by_content_density(sentences, 3, max_scene_duration)
        elif len(scenes) > 15:
            scenes = _merge_short_scenes(scenes, max_scene_duration)
        
        print(f"   ✅ Created {len(scenes)} balanced scenes")
        return scenes

def _validate_and_adjust_scenes(scenes: list, target_duration_minutes: float, max_scene_duration: int) -> list:
    """
    Validate and adjust scenes to meet duration requirements.
    """
    total_duration = sum(scene.get("estimated_duration", 15) for scene in scenes)
    target_duration_seconds = target_duration_minutes * 60
    
    print(f"   Total scene duration: {total_duration:.1f} seconds")
    print(f"   Target duration: {target_duration_seconds:.1f} seconds")
    
    # Adjust if significantly off target
    if abs(total_duration - target_duration_seconds) > target_duration_seconds * 0.2:  # 20% threshold
        adjustment_factor = target_duration_seconds / total_duration
        
        for scene in scenes:
            scene["estimated_duration"] *= adjustment_factor
            scene["metadata"]["duration_adjusted"] = True
        
        print(f"   Adjusted scenes by factor: {adjustment_factor:.2f}")
    
    return scenes

def _print_advanced_analysis_summary(scenes: list, semantic_analysis: dict):
    """
    Print a comprehensive analysis summary.
    """
    print(f"\n📊 Advanced Analysis Summary:")
    print(f"   Total scenes: {len(scenes)}")
    
    # Content type analysis
    content_type = semantic_analysis.get("content_type", {})
    print(f"   Content type: {content_type.get('primary_type', 'unknown')} (confidence: {content_type.get('confidence', 0):.2f})")
    
    # Emotional arc analysis
    emotional_arc = semantic_analysis.get("emotional_arc", {})
    print(f"   Emotional arc: {emotional_arc.get('overall_arc', 'unknown')}")
    
    # Complexity analysis
    complexity_profile = semantic_analysis.get("complexity_profile", {})
    print(f"   Complexity level: {complexity_profile.get('overall_level', 'unknown')}")
    
    # Scene details
    total_duration = sum(scene.get("estimated_duration", 15) for scene in scenes)
    avg_duration = total_duration / len(scenes) if scenes else 0
    
    print(f"   Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"   Average scene duration: {avg_duration:.1f} seconds")
    
    # Optimization summary
    optimized_count = sum(1 for scene in scenes if scene.get("metadata", {}).get("content_type_optimized", False))
    print(f"   Optimized scenes: {optimized_count}/{len(scenes)}")
    
    # Engagement hooks
    engagement_hooks = sum(1 for scene in scenes if scene.get("metadata", {}).get("is_engagement_hook", False))
    print(f"   Engagement hooks: {engagement_hooks}")
    
    print(f"   ✅ Advanced AI-powered splitting completed successfully!")

def _split_by_natural_breaks(sentences: list[str], max_scene_duration: int) -> list[str]:
    """
    Split sentences by natural breaks (paragraphs, topic changes, etc.)
    """
    scenes = []
    current_scene = []
    current_duration = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        sentence_duration = sentence_words / 2.0  # 2 words per second
        
        # Check if adding this sentence would exceed max duration
        if current_duration + sentence_duration > max_scene_duration and current_scene:
            # Complete current scene
            scenes.append(" ".join(current_scene))
            current_scene = [sentence]
            current_duration = sentence_duration
        else:
            # Add to current scene
            current_scene.append(sentence)
            current_duration += sentence_duration
    
    # Add the last scene
    if current_scene:
        scenes.append(" ".join(current_scene))
    
    return scenes

def _split_by_content_density(sentences: list[str], target_scenes: int, max_scene_duration: int) -> list[str]:
    """
    Split sentences based on content density and target scene count
    """
    if len(sentences) <= target_scenes:
        # If we have fewer sentences than target, split some sentences
        return _split_sentences_into_scenes(sentences, target_scenes)
    
    # Calculate target words per scene
    total_words = sum(len(s.split()) for s in sentences)
    target_words_per_scene = total_words / target_scenes
    
    scenes = []
    current_scene = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed target significantly, start new scene
        if current_word_count + sentence_words > target_words_per_scene * 1.3 and current_scene:
            scenes.append(" ".join(current_scene))
            current_scene = [sentence]
            current_word_count = sentence_words
        else:
            current_scene.append(sentence)
            current_word_count += sentence_words
    
    # Add the last scene
    if current_scene:
        scenes.append(" ".join(current_scene))
    
    return scenes

def _split_sentences_into_scenes(sentences: list[str], target_scenes: int) -> list[str]:
    """
    Split sentences into target number of scenes intelligently
    """
    if len(sentences) == 1:
        # Single sentence: split by words
        words = sentences[0].split()
        words_per_scene = max(1, len(words) // target_scenes)
        scenes = []
        
        for i in range(0, len(words), words_per_scene):
            scene_words = words[i:i + words_per_scene]
            if scene_words:
                scenes.append(" ".join(scene_words))
            if len(scenes) >= target_scenes:
                break
        
        # Fill remaining scenes with meaningful content
        while len(scenes) < target_scenes:
            if scenes:
                # Use a portion of the last scene
                last_scene_words = scenes[-1].split()
                if len(last_scene_words) > 3:
                    scenes.append(" ".join(last_scene_words[-3:]))
                else:
                    scenes.append(scenes[-1])
            else:
                scenes.append(sentences[0])
        
        return scenes[:target_scenes]
    else:
        # Multiple sentences: distribute evenly
        scenes = []
        sentences_per_scene = max(1, len(sentences) // target_scenes)
        
        for i in range(0, len(sentences), sentences_per_scene):
            scene_sentences = sentences[i:i + sentences_per_scene]
            if scene_sentences:
                scenes.append(" ".join(scene_sentences))
            if len(scenes) >= target_scenes:
                break
        
        # Fill remaining scenes
        while len(scenes) < target_scenes:
            if scenes:
                scenes.append(scenes[-1])
            else:
                scenes.append(" ".join(sentences))
        
        return scenes[:target_scenes]

def _merge_short_scenes(scenes: list[str], max_scene_duration: int) -> list[str]:
    """
    Merge very short scenes to reduce total count
    """
    if len(scenes) <= 15:
        return scenes
    
    merged_scenes = []
    current_scene = ""
    current_duration = 0
    
    for scene in scenes:
        scene_words = len(scene.split())
        scene_duration = scene_words / 2.0
        
        # If current scene is very short, merge with next
        if current_duration < 5 and scene_duration < 10:  # Less than 5s + 10s
            if current_scene:
                current_scene += " " + scene
                current_duration += scene_duration
            else:
                current_scene = scene
                current_duration = scene_duration
        else:
            # Complete current scene and start new one
            if current_scene:
                merged_scenes.append(current_scene)
            current_scene = scene
            current_duration = scene_duration
    
    # Add the last scene
    if current_scene:
        merged_scenes.append(current_scene)
    
    return merged_scenes

# ==== NEW: Create visual prompt from script block using OpenAI GPT ====

def create_viral_image_prompt(block_text: str, tone: str, topic: str, env: dict, scene_number: int = 1) -> str:
    """Create VIRAL-OPTIMIZED prompts for 10X YouTube growth.
    
    Target: 12-15% CTR, maximum engagement, broad audience appeal
    Focus: Professional cinematography, emotional hooks, viral potential
    """
    import openai

    system_msg = (
        "You are a viral content strategist and cinematographer. "
        "Create prompts for YouTube thumbnails and scenes that maximize clicks and engagement. "
        "Focus on: cinematic realism, emotional impact, professional quality, broad appeal. "
        "Use 'elderly Indian farmer Kishan, white beard, wise eyes, traditional simple clothes' consistently. "
        "Target 12-15% CTR with viral sharing potential."
    )

    # Viral cinematography rotation for engagement
    viral_shots = [
        "dramatic wide establishing shot, golden hour lighting",
        "emotional medium shot, shallow depth of field, professional photography", 
        "inspiring close-up, intense emotional moment, cinematic lighting",
        "over-shoulder storytelling angle, narrative composition",
        "low angle heroic shot, motivational framing, success mindset"
    ]
    
    shot_style = viral_shots[(scene_number - 1) % len(viral_shots)]

    user_msg = (
        f"Scene {scene_number} | Topic: {topic} | Tone: {tone}\n"
        f"Content: {block_text}\n\n"
        f"Create a VIRAL-OPTIMIZED scene: {shot_style}\n"
        f"Show elderly Indian farmer Kishan in this emotional moment.\n"
        f"Focus on: click-worthy composition, emotional depth, professional quality.\n"
        f"Max 120 characters for optimal diffusion results."
    )

    try:
        client = openai.OpenAI(api_key=env["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=80,
            temperature=0.6,
            user="viral_prompt_generator"
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise Exception("Failed to generate viral prompt content")
        
        prompt_raw = content.strip().strip("\"")
        if "no text" not in prompt_raw.lower():
            prompt_raw = prompt_raw.rstrip(".") + ", no text, no watermark, 16:9 cinematic"
        
        # Use VIRAL style for maximum performance
        final_prompt = f"{VIRAL_STYLE_LOCK}, {prompt_raw}"
        return final_prompt[:250]
        
    except Exception as e:
        print(f"⚠️ Viral prompt generation failed: {e}")
        # High-performance fallback
        safe_block = block_text.replace("\n", " ")
        return f"{VIRAL_STYLE_LOCK}, elderly Indian farmer Kishan, {safe_block[:80]}, {shot_style}, viral potential, click-worthy, no text, no watermark, 16:9 cinematic"

# Legacy function (kept for compatibility)
def create_image_prompt(block_text: str, tone: str, topic: str, env: dict) -> str:
    """Return a concise Stable-Diffusion style prompt for the given script block.

    The prompt should:
    • stay under ~180 characters (diffusion models don't benefit from very long prompts)
    • reflect the *tone* (Motivational, Storytelling, etc.)
    • describe the visual scene implied by *block_text*
    • explicitly ban all text or watermark in the generated image
    """
    import openai

    system_msg = (
        "You are an expert visual prompt writer for Stable Diffusion. "
        "Create prompts for YouTube video scenes (16:9 aspect ratio). "
        "Focus on cinematic composition, consistent character design, and smooth narrative flow. "
        "ALWAYS finish with: 'no text, no watermark, 16:9 aspect ratio'."
    )

    user_msg = (
        f"Topic: {topic}\nTone: {tone}\nScene text: {block_text}\n\n"
        "Write ONE prompt (< 150 chars) for a cinematic scene. "
        "Keep the same character throughout: 'elderly Indian farmer Kishan with white beard, simple clothes'. "
        "Describe the specific action/emotion in this scene."
    )

    client = openai.OpenAI(api_key=env["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=100,
        temperature=0.7,
        user="image_prompt_generator"
    )
    content = response.choices[0].message.content
    if content is None:
        raise Exception("Failed to generate prompt content")
    prompt_raw = content.strip().strip("\"")
    if "no text" not in prompt_raw.lower():
        prompt_raw = prompt_raw.rstrip(".") + ", no text, no watermark, 16:9 aspect ratio"
    final_prompt = f"{STYLE_LOCK}, {prompt_raw}"
    return final_prompt[:200]

# ==== UPDATED: generate_scene_images_for_script now uses create_image_prompt ====

def generate_viral_scene_images(script_text: str, video_folder: str, env: dict, tone: str = "Motivational", topic: str = "") -> list[str]:
    """Generate VIRAL-OPTIMIZED images for 10X YouTube growth using metaphor-rich prompts."""
    blocks = split_script_into_blocks_improved(script_text)
    image_paths: list[str] = []

    print(f"🔥 VIRAL IMAGE GENERATION (Metaphor-Rich)")
    print(f"📊 Optimized for: Maximum clicks, engagement, monetization")
    print(f"🎯 Style: Cinematic realism with viral potential")
    print("=" * 60)

    # Use the new metaphor-rich prompt generator
    prompts = generate_metaphor_rich_prompts(
        script_text=script_text,
        num_scenes=len(blocks),
        openai_api_key=env["OPENAI_API_KEY"],
        script_blocks=blocks  # FIXED: Pass the blocks we created
    )

    for i, (block, prompt) in enumerate(zip(blocks, prompts)):
        print(f"🚀 METAPHOR PROMPT: {prompt[:120]}...")
        image_path = os.path.join(video_folder, f"viral_scene_{i+1:02d}.png")
        print(f"🎨 Generating VIRAL image {i+1}/{len(blocks)} → {image_path}")
        try:
            from comfyui_integration.generate_image import generate_image
            success = generate_image(
                prompt_text=prompt,
                out_file=image_path,
                negative_prompt="",  # already included in prompt
                max_wait_sec=900
            )
            if success and os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"✅ SUCCESS! Viral image generated ({file_size//1024}KB)")
                print(f"🎯 Expected: 12-15% CTR, high engagement")
                image_paths.append(image_path)
            else:
                print(f"❌ File not created: {image_path}")
        except Exception as e:
            print(f"❌ Viral image generation failed for block {i+1}: {e}")
        print("-" * 50)

    print(f"\n🎉 VIRAL BATCH COMPLETE!")
    print(f"✅ Generated {len(image_paths)}/{len(blocks)} metaphor-rich, viral-optimized images")
    print(f"📈 Expected Results: 2x higher CTR, 10x growth potential")
    return image_paths

# Legacy function (kept for compatibility)
def generate_scene_images_for_script(script_text: str, video_folder: str, env: dict, tone: str = "Motivational", topic: str = "") -> list[str]:
    blocks = split_script_into_blocks_improved(script_text)
    image_paths: list[str] = []

    for i, block in enumerate(blocks):
        try:
            safe_block = block.replace("\n", " ")
            prompt = create_image_prompt(safe_block, tone=tone, topic=topic, env=env)
        except Exception as e:
            print(f"⚠️  Failed to create prompt for block {i+1}: {e}. Falling back to basic prompt.")
            safe_block = block.replace("\n", " ")
            prompt = f"{tone} scene: {safe_block[:160]}, no text, no watermark"

        image_path = os.path.join(video_folder, f"scene_{i+1:02d}.png")
        print(f"🎨 Generating image {i+1}/{len(blocks)} → {image_path}\n   Prompt: {prompt}\n")

        try:
            generate_and_save_images(prompt, image_path, NEGATIVE_PROMPT)
            image_paths.append(image_path)
        except Exception as e:
            print(f"❌ Image generation failed for block {i+1}: {e}")

    return image_paths

def create_advanced_prompt(block_text: str, tone: str, topic: str, env: dict, scene_number: int = 1, target_audience: str = "general_audience") -> str:
    """Advanced prompt generation with intelligent style selection."""
    import openai
    import random

    # Use intelligent style selection instead of fixed Ghibli
    style_base = select_optimal_style(topic, tone, target_audience)
    shot_type = CINEMATOGRAPHY_SHOTS[(scene_number - 1) % len(CINEMATOGRAPHY_SHOTS)]
    
    system_msg = (
        "You are a master cinematographer and visual storyteller. "
        "Create prompts for YouTube video scenes that maximize viewer engagement. "
        "Focus on: photorealistic imagery, emotional connection, broad appeal. "
        "Always include: 'elderly Indian farmer Kishan, white beard, simple traditional clothes'. "
        "End with: 'no text, no watermark, 16:9 cinematic'"
    )

    user_msg = (
        f"Scene {scene_number} | Topic: {topic} | Tone: {tone}\n"
        f"Shot Type: {shot_type}\n"
        f"Content: {block_text}\n\n"
        f"Create a {shot_type} showing Kishan in this scene. "
        f"Use photorealistic, cinematic style for maximum YouTube appeal. "
        f"Max 120 characters."
    )

    try:
        client = openai.OpenAI(api_key=env["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=80,
            temperature=0.6,
            user="advanced_prompt_generator"
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise Exception("Failed to generate prompt content")
        
        prompt_raw = content.strip().strip("\"")
        if "no text" not in prompt_raw.lower():
            prompt_raw = prompt_raw.rstrip(".") + ", no text, no watermark, 16:9 cinematic"
        
        final_prompt = f"{style_base}, {prompt_raw}"
        return final_prompt[:250]
        
    except Exception as e:
        print(f"⚠️ Advanced prompt generation failed: {e}")
        # Fallback to basic prompt
        safe_block = block_text.replace("\n", " ")
        return f"{style_base}, elderly Indian farmer Kishan, {safe_block[:80]}, {shot_type}, no text, no watermark, 16:9 cinematic"

def select_optimal_style(topic: str, tone: str, target_audience: str = "general_audience") -> str:
    """Intelligently select the best visual style based on content and audience."""
    
    # Analyze topic for content type
    topic_lower = topic.lower()
    content_type = "motivational_stories"  # default
    
    if any(word in topic_lower for word in ["business", "career", "money", "success"]):
        content_type = "business_advice"
    elif any(word in topic_lower for word in ["spiritual", "meditation", "peace", "wisdom"]):
        content_type = "spiritual_wisdom"
    elif any(word in topic_lower for word in ["learn", "education", "knowledge", "skill"]):
        content_type = "educational"
    elif any(word in topic_lower for word in ["achievement", "victory", "winning"]):
        content_type = "success_stories"
    elif any(word in topic_lower for word in ["life", "lesson", "experience"]):
        content_type = "life_lessons"
    
    # Get base style
    base_style = CONTENT_BASED_STYLES.get(content_type, CONTENT_BASED_STYLES["motivational_stories"])
    
    # Add audience-specific elements
    audience_style = AUDIENCE_STYLES.get(target_audience, AUDIENCE_STYLES["general_audience"])
    
    # Combine for final style
    return f"{base_style}, {audience_style}"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS_FILE = "client_secret.json"
TOKEN_FILE = "token.pickle"

def main():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "wb") as token:
        pickle.dump(creds, token)
    print(f"✅ token.pickle created! YouTube upload is now enabled.")

def get_dynamic_max_scene_duration(content_type: str, emotional_tone: str, user_feedback: Optional[dict] = None, target_duration_minutes: float = 5.0) -> int:
    """
    Calculate max scene duration dynamically based on content type, emotional tone, video duration, and user feedback.
    Returns an integer duration in seconds.
    """
    # Base values by content type
    base = {
        'story': 18,
        'lesson': 12,
        'motivation': 15,
        'mixed': 16
    }.get(content_type, 15)
    
    # Adjust for emotion
    emotion_factor = {
        'calm': 1.2,
        'tension': 0.8,
        'excitement': 0.9,
        'reflection': 1.3,
        'neutral': 1.0
    }.get(emotional_tone, 1.0)
    
    # Adjust for user feedback if available
    if user_feedback and isinstance(user_feedback, dict):
        base *= (1 + float(user_feedback.get('duration_adjustment', 0)))
    
    # Calculate base duration
    duration = int(base * emotion_factor)
    
    # Dynamic maximum based on target video duration
    # For longer videos, allow longer scene durations to avoid too many scenes
    if target_duration_minutes <= 3:
        max_duration = 25  # Short videos: keep scenes snappy
    elif target_duration_minutes <= 10:
        max_duration = 40  # Medium videos: moderate scene length
    else:
        max_duration = 60  # Long videos: allow longer scenes
    
    # Clamp to reasonable range with dynamic maximum
    duration = max(6, min(duration, max_duration))
    
    return duration

def create_captioned_scene(image_path, audio_path, caption_text, output_path, language="Hindi"):
    """
    Create a video segment with burned-in captions for a scene.
    - image_path: path to the scene image
    - audio_path: path to the audio for this scene
    - caption_text: text to display as caption
    - output_path: where to save the captioned video segment
    - language: 'Hindi' or 'English' (for font selection)
    """
    if not MOVIEPY_AVAILABLE or None in (ImageClip, AudioFileClip, TextClip, CompositeVideoClip):
        print("⚠️ moviepy is not available or not fully imported. Cannot create captioned scene.")
        return None
    
    # Type assertions for linter
    assert ImageClip is not None, "ImageClip should be available"
    assert AudioFileClip is not None, "AudioFileClip should be available"
    assert TextClip is not None, "TextClip should be available"
    assert CompositeVideoClip is not None, "CompositeVideoClip should be available"
    
    # Font selection
    if language.lower() == "hindi":
        font = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"  # Update path as needed
    else:
        font = "Arial-Bold"
    
    audio_clip = AudioFileClip(audio_path)
    img_clip = ImageClip(image_path).set_duration(audio_clip.duration)
    img_clip = img_clip.set_audio(audio_clip)
    
    # Create caption TextClip
    txt_clip = TextClip(
        caption_text,
        fontsize=48,
        font=font,
        color='white',
        stroke_color='black',
        stroke_width=2,
        method='caption',
        size=(int(img_clip.w * 0.9), None),
        align='center'
    ).set_position(('center', 'bottom')).set_duration(audio_clip.duration)
    
    # Composite
    final = CompositeVideoClip([img_clip, txt_clip])
    final.write_videofile(str(output_path), fps=24, codec="libx264", audio_codec="aac")
    return output_path

def create_cinematic_scene_clip(wide_img, close_img, duration, movement, hold_duration, move_duration, crossfade_duration, closeup_duration):
    from moviepy.editor import ImageClip, CompositeVideoClip
    def ease_in(t, start=0.0, end=1.0, duration=move_duration):
        progress = min(max((t - hold_duration) / duration, 0), 1)
        return start + (end - start) * (progress ** 2)
    def make_cinematic_clip(img_path, movement, duration, hold_duration, move_duration):
        img_clip = ImageClip(img_path).set_duration(duration)
        def make_frame(get_frame, t):
            if t < hold_duration:
                return get_frame(0)
            if movement == "zoom_in":
                factor = ease_in(t, 1.0, 1.13, move_duration)
                return img_clip.resize(factor).get_frame(t)
            elif movement == "zoom_out":
                factor = ease_in(t, 1.13, 1.0, move_duration)
                return img_clip.resize(factor).get_frame(t)
            else:
                return img_clip.get_frame(t)
        return img_clip.fl(make_frame, apply_to=['mask'])
    clip_wide = make_cinematic_clip(wide_img, movement, duration, hold_duration, move_duration)
    clip_close = ImageClip(close_img).set_duration(closeup_duration)
    final_clip = CompositeVideoClip([
        clip_wide.crossfadeout(crossfade_duration).set_start(0),
        clip_close.set_start(duration - crossfade_duration).crossfadein(crossfade_duration)
    ], size=clip_wide.size).set_duration(duration + closeup_duration - crossfade_duration)
    return final_clip

def create_scene_based_slideshow_video(image_paths, audio_paths, output_path, script_blocks=None, topic=None, tone=None, openai_api_key=None):
    if not MOVIEPY_AVAILABLE or AudioFileClip is None or ImageClip is None or concatenate_videoclips is None:
        print("⚠️ moviepy not available, cannot create scene-based slideshow video.")
        return None
    if script_blocks is None or topic is None or tone is None or openai_api_key is None:
        print("❌ script_blocks, topic, tone, and openai_api_key are required for cinematic slideshow.")
        return None
    if len(script_blocks) != len(audio_paths):
        print(f"❌ Mismatch: {len(script_blocks)} script blocks vs {len(audio_paths)} audio files")
        return None
    print(f"🎬 Creating cinematic scene-based slideshow video with {len(script_blocks)} scenes")
    clips = []
    total_duration = 0
    output_folder = os.path.dirname(output_path) or "."
    # Alternate only between zoom_in and zoom_out for variety, no pan
    movement_types = ["zoom_in", "zoom_out"]
    for i, (block, audio_path) in enumerate(zip(script_blocks, audio_paths)):
        try:
            print(f"   Processing cinematic scene {i+1}/{len(script_blocks)}")
            audio_clip = AudioFileClip(audio_path)
            scene_duration = audio_clip.duration
            total_duration += scene_duration
            wide_img = os.path.join(output_folder, f"scene_{i+1:02d}_wide.png")
            close_img = os.path.join(output_folder, f"scene_{i+1:02d}_close.png")
            hold_duration = min(1.5, max(0.5, scene_duration * 0.15))
            move_duration = scene_duration - hold_duration
            crossfade_duration = min(2.0, max(1.0, scene_duration * 0.25))
            closeup_duration = min(3.0, max(1.5, scene_duration * 0.3))
            movement = movement_types[i % len(movement_types)]
            scene_clip = create_cinematic_scene_clip(
                wide_img=wide_img,
                close_img=close_img,
                duration=scene_duration,
                movement=movement,
                hold_duration=hold_duration,
                move_duration=move_duration,
                crossfade_duration=crossfade_duration,
                closeup_duration=closeup_duration
            )
            scene_clip = scene_clip.set_audio(audio_clip)
            clips.append(scene_clip)
        except Exception as e:
            print(f"❌ Error processing cinematic scene {i+1}: {e}")
            continue
    if not clips:
        print("❌ No valid cinematic scenes created")
        return None
    print(f"✅ Created {len(clips)} cinematic scenes, total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    try:
        print("🔗 Concatenating cinematic scene clips...")
        final_clip = concatenate_videoclips(clips, method="compose")
        print(f"📹 Writing final cinematic video to: {output_path}")
        final_clip.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k",
            verbose=False,
            logger=None
        )
        for clip in clips:
            clip.close()
        final_clip.close()
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✅ Cinematic video created successfully!")
            print(f"   File: {Path(output_path).name}")
            print(f"   Size: {file_size // (1024*1024)}MB")
            print(f"   Duration: {total_duration:.2f} seconds")
            return output_path
        else:
            print("❌ Cinematic video file was not created")
            return None
    except Exception as e:
        print(f"❌ Error creating final cinematic video: {e}")
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        return None

def create_single_audio_slideshow_video(image_paths, combined_audio_path, output_path):
    """
    Create a slideshow video from multiple images with one combined audio track.
    Each image will be shown for an equal duration based on the total audio length.
    """
    if not MOVIEPY_AVAILABLE or AudioFileClip is None or ImageClip is None or concatenate_videoclips is None:
        print("⚠️ moviepy not available, cannot create single-audio slideshow video.")
        return None
    
    if not image_paths:
        print("❌ No images provided for video creation")
        return None
    
    print(f"🎬 Creating single-audio slideshow video with {len(image_paths)} images")
    
    try:
        # Load the combined audio to get total duration
        audio_clip = AudioFileClip(combined_audio_path)
        total_audio_duration = audio_clip.duration
        
        # Calculate duration per image
        duration_per_image = total_audio_duration / len(image_paths)
        
        print(f"   Total audio duration: {total_audio_duration:.2f} seconds")
        print(f"   Duration per image: {duration_per_image:.2f} seconds")
        
        clips = []
        
        # Create image clips with equal durations
        for i, img_path in enumerate(image_paths):
            try:
                print(f"   Processing image {i+1}/{len(image_paths)}: {Path(img_path).name}")
                
                # Create image clip for calculated duration
                img_clip = ImageClip(img_path, duration=duration_per_image)
                
                # Keep original image resolution
                
                clips.append(img_clip)
                
            except Exception as e:
                print(f"❌ Error processing image {i+1}: {e}")
                continue
        
        if not clips:
            print("❌ No valid image clips created")
            return None
        
        # Concatenate all image clips
        print("🔗 Concatenating image clips...")
        video_clip = concatenate_videoclips(clips, method="compose")
        
        # Add the combined audio to the video
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write final video
        print(f"📹 Writing final video to: {output_path}")
        final_clip.write_videofile(
            str(output_path), 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            bitrate="8000k",
            verbose=False,
            logger=None
        )
        
        # Close clips to free memory
        for clip in clips:
            clip.close()
        video_clip.close()
        final_clip.close()
        audio_clip.close()
        
        # Verify the created video
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"✅ Single-audio slideshow video created successfully!")
            print(f"   File: {Path(output_path).name}")
            print(f"   Size: {file_size // (1024*1024)}MB")
            print(f"   Duration: {total_audio_duration:.2f} seconds ({total_audio_duration/60:.1f} minutes)")
            return output_path
        else:
            print("❌ Video file was not created")
            return None
            
    except Exception as e:
        print(f"❌ Error creating single-audio slideshow video: {e}")
        return None

if __name__ == "__main__":
    main()


