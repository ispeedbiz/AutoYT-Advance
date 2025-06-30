# pyright: reportMissingImports=false
# main.py — Fully automated pipeline for generating and uploading YouTube videos
import os, sys, time, subprocess, traceback, requests, io, textwrap, logging, pickle, random
from datetime import datetime
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import ImageFont, ImageDraw, Image
import openai
import io
import random
from pathlib import Path
import openai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from prompt_utils import generate_scene_plan_and_prompts, generate_simple_visual_prompts
from utils import (
    load_env, get_video_folder, save_text_to_file, generate_thumbnail,
    sanitize_folder_name, split_script_into_blocks, generate_ai_image,
    make_slideshow_video, generate_images_from_script, text_to_mp3_block, IMAGE_STYLES,
    generate_scene_images_for_script, split_script_into_blocks_improved, generate_viral_scene_images,
    generate_metaphor_rich_prompts, split_script_by_duration
)

# Import new advanced features
try:
    from video_interpolation import create_interpolated_video, VideoInterpolator
    VIDEO_INTERPOLATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: video_interpolation import failed: {e}")
    VIDEO_INTERPOLATION_AVAILABLE = False
    create_interpolated_video = None
    VideoInterpolator = None

try:
    from llm_video_generator import generate_video_plan, LLMVideoGenerator
    LLM_VIDEO_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: llm_video_generator import failed: {e}")
    LLM_VIDEO_GENERATION_AVAILABLE = False
    generate_video_plan = None
    LLMVideoGenerator = None

try:
    from automated_feedback_loop import create_feedback_loop, AutomatedFeedbackLoop
    AUTOMATED_FEEDBACK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: automated_feedback_loop import failed: {e}")
    AUTOMATED_FEEDBACK_AVAILABLE = False
    create_feedback_loop = None
    AutomatedFeedbackLoop = None

# Conditional import for moviepy to handle Python 3.13 compatibility issues
try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: moviepy import failed: {e}")
    print("💡 Video processing features will be limited")
    MOVIEPY_AVAILABLE = False
    ImageClip = None
    AudioFileClip = None
    concatenate_videoclips = None

# Conditional import for pydub to handle potential import issues
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: pydub import failed: {e}")
    print("💡 Audio processing features will be limited")
    print("💡 This is likely due to Python 3.13 compatibility issues with pyaudioop")
    PYDUB_AVAILABLE = False
    AudioSegment = None

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
SERVICE_ACCOUNT_FILE = "service_account.json"
SHEET_NAME = "Back to Zero – Input Sheet"
WORKSHEET_INDEX = 0
SUPPORTED_LANGUAGES = {"Hindi"}  # Only Hindi is fully supported out-of-the-box
BACKGROUND_DIR = BASE_DIR / "backgrounds"
ATTRIBUTED_AUDIO = "Pensive Piano - Audionautix.mp3"

# Setup logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def sheet_ws():
    # Authenticate Google Sheets client
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ])
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).get_worksheet(WORKSHEET_INDEX)

def parse_publish_datetime(s):
    try:
        s = s.strip()
        if len(s) == 10:  # if only date provided, append midnight time
            s += " 00:00"
        return datetime.strptime(s, "%Y-%m-%d %H:%M")
    except:
        return None

def list_pending(ws):
    now = datetime.now()
    rows = ws.get_all_values()
    # Return list of tuples (sheet_row_number, row_data) for rows that are waiting and due
    return [
        (i+2, r)  # i+2 because sheet rows are 1-indexed and include header
        for i, r in enumerate(rows[1:])  # skip header
        if (r[5].strip().lower() in ("", "🕓 waiting") and (pub_date := parse_publish_datetime(r[4])) is not None and pub_date <= now)
    ]

def batch_update(ws, row_num, status, link="", err="", proc_time=""):
    # Update the Google Sheet cells for the given row (F:J columns) with status, link, timestamp, error, proc_time
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.batch_update([{
        "range": f"F{row_num}:J{row_num}",
        "values": [[status, link, ts, err, proc_time]]
    }])

def row_to_data(row):
    # Map a spreadsheet row (list of strings) to a data dict expected by the pipeline
    def g(idx, default=""):
        return row[idx].strip() if idx < len(row) and row[idx].strip() != "" else default
    return {
        "topic": g(0),
        "lang": g(1).capitalize(),
        "title": g(2),
        "notes": g(3),
        "pub_date": g(4),
        "thumb_txt": g(10, g(2, g(0))),
        "tone": g(11, "Motivational"),
        "highlights": g(12, ""),
        "audience": g(13, ""),
        "cta": g(14, "Subscribe"),
        "duration": g(15, ""),
        "thumb_src": g(16, "AUTO"),
        "attrib": g(17, ""),
        "imageurl": g(18, "")
    }

# ─────────────────── Prompt builder for script generation ───────────────────
def build_prompt(d):
    """
    Builds a conversational and emotionally engaging narration prompt for Hindi YouTube scripts.
    """
    return textwrap.dedent(f"""
    तुम एक soulful YouTube narrator हो, जिसकी आवाज़ में गहराई, connection और सादगी होनी चाहिए —
    Style 'Syllabus with Rohit' से inspired हो, लेकिन delivery तुम्हारी unique हो: {d['tone']}, honest, और real.

    🔹 Language: Mainly Hindi — but keep English words like "AI", "mindset", "Deep Work", "Finance" as-is.  
    🔹 Length: {d['duration']} (script ≤ 600 seconds)  
    🔹 Audience: {d['audience']}  
    🔹 Topic: "{d['topic']}"

    — FLOW (Only for internal guidance, don't include labels) —
    1. Start with hook as a bold question or emotional or curious or thought provoking hook (≤ 30 words).  
    2. Give 1-2 line context why it matters today.
    3. Deliver 3 punchy takeaways (≤ 60 words), including these if relevant: {d['highlights']}  
    4. Share quick Indian story/unique ancient story/philosophers story/analogy (≤ 120 words)
    5. Philosophical reflection on the topic (≤ 30 words).  
    6. Ask one deep reflection question to the viewer.  
    7. End with a warm CTA: "{d['cta']}" — simple, friendly, and inviting.

    — DELIVERY STYLE —
    • Speak slowly, softly, emotionally, calmly — like a mentor or friend.  
    • Use short lines (≤ 20 words), keep rhythm.  
    • Add a 1-second pause after each line.  
    • Directly speak to the viewer: "सोचिए", "आपने कभी महसूस किया?", "imagine कीजिए…", any other similar phrases.  
    • Use Hindi as base, but don't translate common English words — use naturally.  
    • Use active voice, deep tone, and authentic emotion.  
    • No labels, headings, quotes, or bullet points in output.  
    • Avoid clichés or filler. Keep it personal and grounded.

    Notes (if any): {d['notes'] or "None"}

    👉 Return only the final narration — no formatting, no structure hints, no extra markup.
    """).strip()

def split_sentences(text):
    # Split script into sentences (approx) for TTS, treating ., ?, !, and Hindi danda (।) as breaks.
    return [s.strip() for s in text.replace("।", ".").replace("?", ".").replace("!", ".").split(".") if s.strip()]

def generate_script(prompt, env, folder):
    # Generate narration script using OpenAI GPT (ChatCompletion)
    import openai
    client = openai.OpenAI(api_key=env["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a calm, emotional Hindi storyteller who narrates emotionally engaging content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.55,
        user="backtozero_youtube"
    )
    text = response.choices[0].message.content
    if text is None:
        raise Exception("Failed to generate script content")
    text = text.strip()
    save_text_to_file(text, folder / "script.txt")
    return text

def text_to_mp3(script, env, folder):
    # Convert text script to speech using ElevenLabs API (or compatible API)
    voice_id = env.get("HINDI_VOICE_ID")
    if not voice_id or voice_id.strip() == "":
            raise Exception("No HINDI_VOICE_ID found in .env or it is blank!")

    if not PYDUB_AVAILABLE:
        print("⚠️ pydub not available, using basic TTS without audio processing")
        # Basic TTS without audio processing
        parts = split_sentences(script)
        output_path = folder / "voice.mp3"
        
        # Just use the first part for basic TTS
        if parts:
            r = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": env["ELEVENLABS_API_KEY"]},
                json={"text": parts[0], "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
            )
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(r.content)
                return output_path
            else:
                raise Exception(f"ElevenLabs API error: {r.status_code} - {r.text}")
        else:
            raise Exception("No text to convert to speech")
    
    # Full pydub-based processing - only executed when PYDUB_AVAILABLE is True
    assert PYDUB_AVAILABLE and AudioSegment is not None, "pydub should be available here"
    
    parts = split_sentences(script)
    output_path = folder / "voice.mp3"
    
    if not parts:
        raise Exception("No text to convert to speech")
    
    # Process each sentence and combine
    combined_audio = AudioSegment.silent(duration=500)  # Start with 0.5s silence
    
    for part in parts:
        if not part.strip():
            continue
            
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": env["ELEVENLABS_API_KEY"]},
            json={"text": part, "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
        )
        
        if r.status_code == 200:
            # Convert response to AudioSegment
            audio_data = io.BytesIO(r.content)
            audio_segment = AudioSegment.from_mp3(audio_data)
            combined_audio += audio_segment + AudioSegment.silent(duration=1000)  # 1s pause between sentences
        else:
            raise Exception(f"ElevenLabs API error: {r.status_code} - {r.text}")
    
    combined_audio.export(output_path, format="mp3")
    return output_path

def add_background(narration_path, folder):
    """
    Overlay a random background music track beneath the narration.
    """
    from music_selector import MusicSelector, DynamicAudioProcessor
    from semantic_analyzer import SemanticAnalyzer
    
    # Path to the final audio file with background music
    output_path = folder / "final_audio.mp3"
    
    try:
        # Initialize music selector and audio processor
        music_selector = MusicSelector(BACKGROUND_DIR)
        audio_processor = DynamicAudioProcessor()
        semantic_analyzer = SemanticAnalyzer()
        
        # Read the script to analyze content for music selection
        script_path = folder / "script.txt"
        if script_path.exists():
            with open(script_path, 'r', encoding='utf-8') as f:
                script = f.read()
            
            # Select music based on content analysis
            selected_music = music_selector.select_music_for_content(script)
            music_file = music_selector.get_music_path(selected_music)
            
            if not music_file:
                # Fallback to random selection
                music_files = list(BACKGROUND_DIR.glob("*.mp3"))
                if not music_files:
                    print("⚠️ No background music files found")
                    return narration_path
                music_file = random.choice(music_files)
                print(f"🎵 Random fallback music: {music_file.name}")
        else:
            # Fallback to random selection if no script file
            music_files = list(BACKGROUND_DIR.glob("*.mp3"))
            if not music_files:
                print("⚠️ No background music files found")
                return narration_path
            music_file = random.choice(music_files)
            print(f"🎵 Random music (no script): {music_file.name}")
        
        print(f"🎵 Adding background music: {music_file.name}")
        
        # Get volume adjustment based on content analysis
        if script_path.exists():
            import re
            sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                analysis = semantic_analyzer.analyze_semantic_structure(sentences)
                emotional_arc = analysis.get("emotional_arc", {}).get("overall_arc", "neutral")
                volume_adjustment = audio_processor.get_volume_adjustment(emotional_arc)
                print(f"   Volume adjustment: {volume_adjustment}dB (based on {emotional_arc} emotion)")
            else:
                volume_adjustment = -18  # Default
        else:
            volume_adjustment = -18  # Default
        
        # Process audio with pydub
        if PYDUB_AVAILABLE and AudioSegment is not None:
            narration_audio = AudioSegment.from_mp3(narration_path)
            bg_music = AudioSegment.from_mp3(music_file) + volume_adjustment  # Apply volume adjustment
            
            # Loop background music to match narration length
            while len(bg_music) < len(narration_audio):
                bg_music = bg_music + bg_music
            
            # Trim background music to exact length
            bg_music = bg_music[:len(narration_audio)]
            
            # Mix audio
            mixed_audio = narration_audio.overlay(bg_music)
            mixed_audio.export(output_path, format="mp3")
            return output_path
        else:
            # Fallback to ffmpeg
            print("🔄 Using ffmpeg fallback for audio mixing")
            music_files = list(BACKGROUND_DIR.glob("*.mp3"))
            if not music_files:
                print("⚠️ No background music files found for ffmpeg fallback")
                return narration_path
            
            music_file = random.choice(music_files)
            print(f"🎵 [ffmpeg fallback] Adding background music: {music_file.name}")
            
            # Use ffmpeg to mix audio
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", str(narration_path),
                "-i", str(music_file),
                "-filter_complex", f"[1:a]volume={volume_adjustment}dB[a1];[0:a][a1]amix=inputs=2:duration=first",
                "-c:a", "mp3",
                str(output_path)
            ]
            subprocess.run(cmd, check=True)
            return output_path
            
    except Exception as e:
        print(f"⚠️ Background music addition failed: {e}")
        return narration_path

def make_video(thumbnail_path, audio_path, folder):
    # Use ffmpeg to create an MP4 video from a single image and an audio file
    video_path = folder / "final_video.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", "2", "-i", str(thumbnail_path),  # use image as a video loop
        "-i", str(audio_path),
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest", str(video_path)
    ], check=True)
    return video_path

def yt_upload(video_path, thumb_path, data):
    # Upload video to YouTube
    creds = pickle.load(open("token.pickle", "rb"))
    youtube = build("youtube", "v3", credentials=creds)
    # Prepare video metadata
    desc = f"{data['topic']} का सारांश\n\nमुख्य बिंदु:\n{data['highlights']}\n\n{data['notes']}\n\n{data['attrib']}"
    body = {
        "snippet": {
            "title": data["title"],
            "description": desc,
            "tags": [data["topic"], data["tone"]],
            "categoryId": "22"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }
    # Upload video file
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while not response:
        _, response = request.next_chunk()
    # Set custom thumbnail
    youtube.thumbnails().set(videoId=response["id"], media_body=MediaFileUpload(str(thumb_path))).execute()
    return f"https://youtu.be/{response['id']}"

def load_feedback_for_script(video_folder, num_scenes):
    feedback_folder = video_folder / "feedback"
    feedbacks = []
    for i in range(num_scenes):
        feedback_path = feedback_folder / f"scene_{i+1:02d}_feedback.txt"
        if feedback_path.exists():
            with open(feedback_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Feedback:"):
                        feedback = line.replace("Feedback:", "").strip()
                        if feedback:
                            feedbacks.append(feedback)
    return feedbacks

def process(row_num, row_data, ws, env):
    import time
    from pathlib import Path

    start_time = time.time()
    try:
        data = row_to_data(row_data)
        video_folder = Path(get_video_folder(VIDEOS_DIR, data["topic"], data["pub_date"]))
        video_folder.mkdir(parents=True, exist_ok=True)

        # Initialize automated feedback loop if available
        feedback_loop = None
        if AUTOMATED_FEEDBACK_AVAILABLE and create_feedback_loop is not None:
            feedback_loop = create_feedback_loop(
                env["OPENAI_API_KEY"], 
                max_attempts=3, 
                quality_threshold=0.7
            )
            print("🔄 Automated feedback loop enabled")

        # 1. Advanced LLM Video Generation (if available)
        if LLM_VIDEO_GENERATION_AVAILABLE and generate_video_plan is not None:
            try:
                print("🎬 Generating advanced video plan with LLM...")
                video_plan = generate_video_plan(
                    topic=data["topic"],
                    tone=data["tone"],
                    duration=data["duration"],
                    audience=data["audience"],
                    highlights=data["highlights"],
                    notes=data["notes"],
                    openai_api_key=env["OPENAI_API_KEY"]
                )
                
                if video_plan:
                    # Save the complete video plan
                    plan_path = video_folder / "video_plan.json"
                    import json
                    with open(plan_path, 'w') as f:
                        json.dump(video_plan, f, indent=2)
                    
                    # Use LLM-generated script if available
                    if "video_script" in video_plan:
                        script_text = video_plan["video_script"].get("hook", "") + "\n\n"
                        for section in video_plan["video_script"].get("sections", []):
                            script_text += section.get("script", "") + "\n\n"
                        script_text += video_plan["video_script"].get("ending", "")
                        
                        # Save LLM-generated script
                        save_text_to_file(script_text, video_folder / "llm_script.txt")
                        print("✅ LLM video plan generated and saved")
                    else:
                        script_text = None
                else:
                    script_text = None
            except Exception as e:
                print(f"⚠️ LLM video generation failed: {e}")
                script_text = None
        else:
            script_text = None

        # 2. Script generation (skip if exists or use LLM-generated script)
        script_path = video_folder / "script.txt"
        language = data.get("lang", "Hindi").strip().capitalize()
        
        if script_path.exists():
            print(f"✅ Script already exists: {script_path}, skipping generation.")
            with open(script_path, 'r', encoding='utf-8') as f:
                script_text = f.read()
        elif script_text:
            # Use LLM-generated script
            save_text_to_file(script_text, script_path)
            print("✅ Using LLM-generated script")
        else:
            # Generate script using traditional method
            if language == "English":
                prompt = textwrap.dedent(f'''
                You are a soulful YouTube narrator with a deep, emotional, and simple voice—
                Inspired by 'Syllabus with Rohit' style, but your delivery is unique: {data['tone']}, honest, and real.

                🔹 Language: English
                🔹 Length: {data['duration']} (script ≤ 600 seconds)
                🔹 Audience: {data['audience']}
                🔹 Topic: "{data['topic']}"

                — FLOW (Only for internal guidance, don't include labels) —
                1. Start with an emotional hook with a quick Indian story/unique ancient story/philosopher's story/analogy (≤ 120 words).
                2. Give 1-2 line context why it matters today.
                3. Deliver 3 punchy takeaways (≤ 60 words), including these if relevant: {data['highlights']}
                4. Share a quick bold question or emotional hook.(≤ 30 words).
                5. Philosophical reflection on the topic (≤ 30 words).
                6. Ask one deep reflection question to the viewer.
                7. End with a warm CTA: "{data['cta']}" — simple, friendly, and inviting.

                — DELIVERY STYLE —
                • Speak slowly, softly, emotionally — like a mentor or friend.
                • Use short lines (≤ 15 words), keep rhythm.
                • Add a 1-second pause after each line.
                • Directly speak to the viewer: "Imagine...", "Have you ever felt...?", or similar phrases.
                • Use active voice, deep tone, and authentic emotion.
                • No labels, headings, quotes, or bullet points in output.
                • Avoid clichés or filler. Keep it personal and grounded.

                Notes (if any): {data['notes'] or "None"}

                👉 Return only the final narration — no formatting, no structure hints, no extra markup.
                ''').strip()
            else:
                prompt = build_prompt(data)
            script_text = generate_script(prompt, env, video_folder)

        # 3. Split script into blocks for image/audio alignment (duration-aware)
        # Parse duration from data to determine optimal scene count
        duration_str = data.get("duration", "5 minutes")
        target_duration_minutes = 5.0  # default
        
        # Extract duration from string (e.g., "2 minutes", "1.5 minutes", "90 seconds")
        import re
        duration_match = re.search(r'(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?)', duration_str.lower())
        if duration_match:
            value = float(duration_match.group(1))
            unit = duration_match.group(2)
            if unit.startswith('sec'):
                target_duration_minutes = value / 60
            else:
                target_duration_minutes = value
        
        print(f"🎬 Target video duration: {target_duration_minutes:.1f} minutes")
        
        # Use duration-aware script splitting
        script_blocks = split_script_by_duration(script_text, target_duration_minutes, max_scene_duration=20)
        num_blocks = len(script_blocks)
        print(f"📝 Script split into {num_blocks} duration-optimized blocks for image/audio alignment.")

        # 4. Generate image prompts for each block (using advanced prompt generator or fallback)
        try:
            image_prompts = generate_metaphor_rich_prompts(
                script_text=script_text,
                num_scenes=len(script_blocks),
                openai_api_key=env["OPENAI_API_KEY"]
            )
            if len(image_prompts) != len(script_blocks):
                print("⚠️ LLM prompt count mismatch, falling back to simple LLM prompts.")
                raise ValueError("Prompt count mismatch")
            for i, (block, prompt) in enumerate(zip(script_blocks, image_prompts)):
                if block.strip() == prompt.strip():
                    print(f"⚠️ Block and prompt are identical for scene {i+1}. Flagging for review.")
        except Exception as e1:
            print(f"⚠️ Advanced prompt generation failed: {e1}")
            # Try simple LLM prompt
            try:
                image_prompts = generate_simple_visual_prompts(
                    script_blocks,
                    openai_api_key=env["OPENAI_API_KEY"]
                )
                if len(image_prompts) != len(script_blocks):
                    print("⚠️ Simple LLM prompt count mismatch, falling back to creative template prompts.")
                    raise ValueError("Prompt count mismatch")
                for i, (block, prompt) in enumerate(zip(script_blocks, image_prompts)):
                    if block.strip() == prompt.strip():
                        print(f"⚠️ Block and prompt are identical for scene {i+1} (simple LLM). Flagging for review.")
            except Exception as e2:
                print(f"⚠️ Simple LLM prompt generation failed: {e2}")
                # Use deterministic creative template
                image_prompts = [f"A metaphorical illustration of: {block[:120]}" for block in script_blocks]
                for i, (block, prompt) in enumerate(zip(script_blocks, image_prompts)):
                    if block.strip() == prompt.strip():
                        print(f"⚠️ Block and prompt are identical for scene {i+1} (template fallback). Flagging for review.")

        # 5. Image generation with automated feedback loop (skip if exists)
        image_paths = []
        images_folder = video_folder / "images"
        prompts_folder = video_folder / "prompts"
        for i, (block, prompt) in enumerate(zip(script_blocks, image_prompts)):
            image_path = images_folder / f"scene_{i+1:02d}.png"
            prompt_path = prompts_folder / f"scene_{i+1:02d}_final_prompt.txt"
            
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(f"Block: {block}\nPrompt: {prompt}\n")
            
            if image_path.exists():
                print(f"✅ Image already exists: {image_path}, skipping generation.")
                image_paths.append(str(image_path))
                continue
            
            # Use automated feedback loop if available
            if feedback_loop:
                print(f"🔄 Generating image {i+1} with automated feedback loop...")
                from comfyui_integration.generate_image import generate_image
                
                def image_generator(prompt_text, output_path):
                    return generate_image(
                        prompt_text=prompt_text,
                        out_file=output_path,
                        negative_prompt="",
                        max_wait_sec=900,
                        quality="ultra"
                    )
                
                success, final_image_path, feedback_data = feedback_loop.run_image_quality_loop(
                    str(image_path), prompt, block, image_generator
                )
                
                # Save feedback report
                feedback_report_path = video_folder / f"scene_{i+1:02d}_feedback_report.json"
                feedback_loop.save_feedback_report(str(feedback_report_path), feedback_data)
                
                if success:
                    print(f"✅ Image {i+1} generated successfully with feedback loop")
                    image_paths.append(str(image_path))
                else:
                    print(f"⚠️ Image {i+1} failed quality threshold after feedback loop")
                    image_paths.append(str(image_path))  # Still use the image
            else:
                # Traditional image generation
                from comfyui_integration.generate_image import generate_image
                success = generate_image(
                    prompt_text=prompt,
                    out_file=str(image_path),
                    negative_prompt="",  # already included in prompt
                    max_wait_sec=900,
                    quality="ultra"  # Use ultra-quality for best results
                )
                
                # Quality check after image generation
                if success and os.path.exists(image_path):
                    try:
                        from quality_filter import QualityFilter
                        quality_filter = QualityFilter(env["OPENAI_API_KEY"])
                        quality_pass, confidence, reasoning, recommendations = quality_filter.comprehensive_quality_check(
                            str(image_path), prompt, block, quality_threshold=0.6
                        )
                        if quality_pass:
                            print(f"✅ Image {i+1} passed quality check (confidence: {confidence:.2f})")
                            image_paths.append(str(image_path))
                        else:
                            print(f"⚠️ Image {i+1} failed quality check (confidence: {confidence:.2f})")
                            print(f"   Issues: {recommendations}")
                            image_paths.append(str(image_path))
                    except Exception as e:
                        print(f"⚠️ Quality check failed for image {i+1}: {e}")
                        image_paths.append(str(image_path))
                else:
                    print(f"❌ Image generation failed for block {i+1}")
            
            # Save feedback file for user/team
            feedback_folder = video_folder / "feedback"
            feedback_folder.mkdir(exist_ok=True)
            feedback_path = feedback_folder / f"scene_{i+1:02d}_feedback.txt"
            with open(feedback_path, 'w', encoding='utf-8') as f:
                f.write(f"Block: {block}\nPrompt: {prompt}\nImage: {image_path}\nFeedback: \n")

        # 6. Audio block generation with automated feedback loop (skip if exists)
        audio_paths = []
        audio_folder = video_folder / "audio"
        for i, block in enumerate(script_blocks):
            audio_out = audio_folder / f"block_{i+1}.mp3"
            if audio_out.exists():
                print(f"✅ Audio block already exists: {audio_out}, skipping TTS.")
                audio_paths.append(str(audio_out))
                continue
            
            # Use automated feedback loop for audio if available
            if feedback_loop:
                print(f"🔄 Generating audio block {i+1} with automated feedback loop...")
                
                def audio_generator(text):
                    return text_to_mp3_block(text, env, audio_out, language=language)
                
                success, final_audio_path, feedback_data = feedback_loop.run_audio_quality_loop(
                    block, audio_generator
                )
                
                # Save feedback report
                feedback_report_path = video_folder / f"audio_block_{i+1}_feedback_report.json"
                feedback_loop.save_feedback_report(str(feedback_report_path), feedback_data)
                
                if success:
                    print(f"✅ Audio block {i+1} generated successfully with feedback loop")
                    audio_paths.append(str(audio_out))
                else:
                    print(f"⚠️ Audio block {i+1} failed quality threshold after feedback loop")
                    audio_paths.append(str(audio_out))  # Still use the audio
            else:
                # Traditional audio generation
                text_to_mp3_block(block, env, audio_out, language=language)
                audio_paths.append(str(audio_out))

        # 7. Combined narration (skip if exists)
        narration_path = audio_folder / "combined_narration.mp3"
        if narration_path.exists():
            print(f"✅ Combined narration already exists: {narration_path}, skipping.")
        else:
            if PYDUB_AVAILABLE:
                assert AudioSegment is not None, "AudioSegment should be available when PYDUB_AVAILABLE is True"
                combined_audio = AudioSegment.silent(duration=500)  # Start with 0.5s silence
                for audio_path in audio_paths:
                    if os.path.exists(audio_path):
                        audio_segment = AudioSegment.from_mp3(audio_path)
                        combined_audio += audio_segment + AudioSegment.silent(duration=1000)  # 1s pause between blocks
                combined_audio.export(narration_path, format="mp3")
            else:
                print("⚠️ pydub not available, using first audio block as narration")
                if audio_paths and os.path.exists(audio_paths[0]):
                    import shutil
                    shutil.copy(audio_paths[0], narration_path)
                else:
                    print("❌ No audio blocks available")
                    return

        # 8. Final audio with background (skip if exists)
        final_audio_path = audio_folder / "final_audio.mp3"
        if final_audio_path.exists():
            print(f"✅ Final audio with background already exists: {final_audio_path}, skipping.")
        else:
            final_audio_path = add_background(narration_path, video_folder)
            print(f"🎵 Background music added: {final_audio_path}")

        # 9. Thumbnail generation (skip if exists)
        thumbnails_folder = video_folder / "thumbnails"
        thumb_final_path = thumbnails_folder / "thumbnail_final.jpg"
        if thumb_final_path.exists():
            print(f"✅ Thumbnail already exists: {thumb_final_path}, skipping.")
        else:
            # Use LLM-generated thumbnail concept if available
            if LLM_VIDEO_GENERATION_AVAILABLE and 'video_plan' in locals() and video_plan:
                thumbnail_concept = video_plan.get("thumbnail_concept", {})
                if thumbnail_concept:
                    thumb_prompt = thumbnail_concept.get("ai_prompt", f"{data['topic']}, bold, clean background, cartoon illustration, vibrant, no text, no watermark")
                else:
                    thumb_prompt = f"{data['topic']}, bold, clean background, cartoon illustration, vibrant, no text, no watermark"
            else:
                thumb_prompt = f"{data['topic']}, bold, clean background, cartoon illustration, vibrant, no text, no watermark"
            
            thumb_img_path = thumbnails_folder / "thumbnail.jpg"
            if not thumb_img_path.exists():
                generate_ai_image(thumb_prompt, env["OPENAI_API_KEY"], thumb_img_path)
            generate_thumbnail(
                title=data['title'],
                img_url=str(thumb_img_path),
                output_path=thumb_final_path
            )

        # 10. Final video with advanced features (skip if exists)
        final_output_folder = video_folder / "final_output"
        video_out = final_output_folder / "final_video.mp4"
        if video_out.exists():
            print(f"✅ Final video already exists: {video_out}, skipping.")
        else:
            # Always use final_audio.mp3 as the audio source
            print(f"🎵 Using audio file for final video: {final_audio_path}")
            # Try video interpolation first if available
            if VIDEO_INTERPOLATION_AVAILABLE and create_interpolated_video is not None:
                print("🎬 Creating video with frame interpolation...")
                success = create_interpolated_video(
                    image_paths=image_paths,
                    audio_path=str(final_audio_path),
                    output_path=str(video_out),
                    interpolation_method="rife"
                )
                if success:
                    print("✅ Video created with frame interpolation")
                else:
                    print("⚠️ Video interpolation failed, falling back to traditional method")
                    if MOVIEPY_AVAILABLE:
                        make_slideshow_video(image_paths, [str(final_audio_path)], video_out)
                    else:
                        print("⚠️ moviepy not available, skipping video creation. Please use Python 3.11/3.12 for full video support.")
            elif MOVIEPY_AVAILABLE:
                make_slideshow_video(image_paths, [str(final_audio_path)], video_out)
            else:
                print("⚠️ moviepy not available, skipping video creation. Please use Python 3.11/3.12 for full video support.")

        # 11. Upload video to YouTube with the custom thumbnail
        video_url = yt_upload(video_out, thumb_final_path, data) if MOVIEPY_AVAILABLE and video_out.exists() else None
        if video_url:
            batch_update(ws, row_num, " ✅ Posted", video_url, "", f"{int(time.time()-start_time)}s")

        # Load feedback for the script
        feedbacks = load_feedback_for_script(video_folder, len(script_blocks))
        # In prompt generation/selection, add:
        # "Previous feedback: {feedbacks}"

    except Exception as e:
        batch_update(ws, row_num, " Error", "", str(e), f"{int(time.time()-start_time)}s")
        logging.error(f"Failed to process row {row_num}: {traceback.format_exc()}")

if __name__ == "__main__":
    env = load_env()
    print("Loaded ENV:", env)

    worksheet = sheet_ws()
    for row_num, row_data in list_pending(worksheet):
        batch_update(worksheet, row_num, "⚙️ Processing")
        process(row_num, row_data, worksheet, env)
