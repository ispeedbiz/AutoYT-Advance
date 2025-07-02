import random
import openai
import os
import re

# Emotion and style vocabularies
EMOTION_WORDS = [
    "nostalgic", "hopeful", "bittersweet", "serene", "triumphant", "melancholic", "joyful", "peaceful", "inspiring", "reflective", "uplifting", "poignant"
]
STYLE_WORDS = [
    "golden hour lighting", "soft focus", "cinematic realism", "wide angle shot", "shallow depth of field", "vivid colors", "dreamy atmosphere", "natural light", "moody shadows", "warm tones"
]

NEGATIVE_PROMPT = (
    "text, watermark, signature, border, frame, lowres, bad anatomy, distorted face, extra limbs, "
    "cartoon, anime, jpeg artifacts, blurry, unprofessional, cropped, cut off, out of frame, "
    "disfigured, deformed, duplicate, repeated, cloned face, multiple heads, multiple faces, "
    "mutated hands, missing fingers, 3d render, sketch, drawing, painting, comic, manga, pixel art, "
    "toy, doll, plastic, statue, CGI, synthetic, harsh shadows, washed out, overexposed, underexposed, "
    "monochrome, sepia, black and white, logo, brand, modern clothes"
)

NEGATIVE_POSES = (
    "meditating, sitting cross-legged, spiritual pose, yoga, praying, "
    "text, watermark, signature, border, frame, lowres, bad anatomy, distorted face, extra limbs, "
    "cartoon, anime, jpeg artifacts, blurry, unprofessional"
)

VISUAL_STYLES = [
    "cinematic realism, golden hour lighting",
    "watercolor painting, soft focus",
    "vintage film, muted colors",
    "high contrast, dramatic shadows",
    "vivid colors, modern digital art",
    "warm tones, dreamy atmosphere"
]

# --- Style lock and shot types for maximum engagement ---
NEGATIVE_PROMPT_LOCK = (
    "no text, no watermark, no extra hands, no extra limbs, no deformed anatomy, "
    "no duplicate faces, no blur, no distortion, no cartoon outline, no anime artifacts, 16:9"
)
SHOT_TYPES = [
    "wide establishing shot",
    "medium shot with shallow depth of field",
    "close-up with emotional focus",
    "over-the-shoulder perspective",
    "low angle heroic shot"
]

# --- Extraction utilities (simple keyword-based for now) ---
def extract_theme(script_block):
    # Simple keyword-based theme extraction
    keywords = ["success", "struggle", "nature", "childhood", "memory", "journey", "family", "village", "river", "market", "growth", "learning", "celebration", "solitude", "hope", "friendship"]
    for word in keywords:
        if word in script_block.lower():
            return word
    return "life"

def extract_emotion(script_block):
    # Simple keyword-based emotion extraction
    emotions = ["joy", "sadness", "hope", "nostalgia", "peace", "triumph", "reflection", "gratitude", "wonder", "longing", "serenity", "inspiration"]
    for word in emotions:
        if word in script_block.lower():
            return word
    return random.choice(EMOTION_WORDS)

def extract_setting(script_block):
    # Simple keyword-based setting extraction
    settings = [
        ("river", "by a peaceful riverbank"),
        ("village", "in a rural Indian village"),
        ("market", "in a bustling village market"),
        ("field", "in a lush green field"),
        ("mountain", "in the foothills of mountains"),
        ("forest", "in a sun-dappled forest"),
        ("home", "inside a humble village home"),
        ("school", "outside a small rural school"),
        ("road", "on a dusty village road"),
        ("tree", "under a large banyan tree"),
        ("temple", "near an ancient temple"),
        ("lake", "beside a tranquil lake"),
        ("garden", "in a blooming garden")
    ]
    for key, desc in settings:
        if key in script_block.lower():
            return desc
    return "in nature"

def extract_time_of_day(script_block):
    # Simple keyword-based time of day extraction
    if any(word in script_block.lower() for word in ["morning", "sunrise", "dawn"]):
        return "at sunrise"
    if any(word in script_block.lower() for word in ["evening", "sunset", "dusk"]):
        return "at sunset"
    if "night" in script_block.lower():
        return "at night"
    if "afternoon" in script_block.lower():
        return "in the afternoon"
    return random.choice(["in soft morning light", "in golden hour", "under a clear sky", "in gentle daylight"])

def detect_person(script_block):
    # Detect if a person is mentioned
    person_words = ["man", "woman", "boy", "girl", "child", "farmer", "mother", "father", "friend", "teacher", "he", "she", "they", "person", "old", "young"]
    return any(word in script_block.lower() for word in person_words)

def extract_action(script_block):
    # Simple action extraction
    actions = ["walking", "sitting", "reading", "working", "smiling", "talking", "thinking", "resting", "looking", "running", "playing", "teaching", "cooking", "meditating", "celebrating", "helping"]
    for word in actions:
        if word in script_block.lower():
            return word
    return random.choice(["standing", "reflecting", "enjoying the moment", "watching the scenery"])

def select_emotion(script_block, emotion_words=EMOTION_WORDS):
    # Use extracted or random emotion
    return extract_emotion(script_block)

def select_style(script_block, style_words=STYLE_WORDS):
    # Randomly select a style
    return random.choice(style_words)

# --- Main enhanced prompt generator ---
def generate_enhanced_prompt(script_block, previous_context=None):
    theme = extract_theme(script_block)
    emotion = select_emotion(script_block)
    setting = extract_setting(script_block)
    time_of_day = extract_time_of_day(script_block)
    contains_person = detect_person(script_block)
    action = extract_action(script_block)
    style = select_style(script_block)

    # Choose template
    if not contains_person:
        template = f"{setting}, {time_of_day}, {emotion}, {theme}, {style}, deeply emotional, cinematic, 16:9"
    else:
        template = f"A person {action} {setting}, {time_of_day}, {emotion}, {theme}, {style}, cinematic, 16:9"

    # Add continuity
    if previous_context:
        template += f", following previous scene: {previous_context}"

    # Add negative prompt
    prompt = f"{template}, no text, no watermark, {NEGATIVE_PROMPT}"
    return prompt[:250]

# --- Engagement scoring for prompts ---
def score_prompt_engagement(prompt, openai_api_key):
    import openai
    import json
    import re
    client = openai.OpenAI(api_key=openai_api_key)
    system_msg = (
        "You are a YouTube visual engagement expert. "
        "Given an AI image prompt, rate its potential for high engagement and virality on a 0-1 scale. "
        "Consider: visual clarity, emotional impact, cinematic appeal, and click-worthiness. "
        "Return ONLY a JSON object: {\"engagement_score\": 0-1, \"reason\": \"...\"} and nothing else."
    )
    user_msg = f"Prompt: {prompt}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=120,
        temperature=0.0,
        user="engagement_scorer"
    )
    content = response.choices[0].message.content
    if not content:
        return 0.0, "No response from engagement scorer"
    try:
        # Try direct JSON parse first
        try:
            result = json.loads(content)
            return float(result.get("engagement_score", 0)), result.get("reason", "")
        except Exception:
            # Fallback: extract JSON substring
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                try:
                    result = json.loads(content[start:end+1])
                    return float(result.get("engagement_score", 0)), result.get("reason", "")
                except Exception:
                    pass
            # Regex fallback for partial/cutoff JSON
            match = re.search(r'\{[^\{\}]*"engagement_score"[^\{\}]*\}', content)
            if match:
                try:
                    result = json.loads(match.group(0))
                    return float(result.get("engagement_score", 0)), result.get("reason", "")
                except Exception:
                    pass
            return 0.0, f"No JSON found in response: {content}"
    except Exception as e:
        return 0.0, f"Parse error: {e} | Raw: {content}"

# --- Enhanced prompt generator with style lock, shot type, negatives, and engagement scoring ---
def generate_metaphor_rich_prompts(script_text, num_scenes, openai_api_key, script_blocks=None, topic=None, tone=None):
    """
    For each script block, use LLM to extract/generate a unique visual metaphor/subject/action,
    then build a descriptive, cinematic prompt for image generation.
    """
    import openai
    import re
    from utils import split_script_into_blocks_improved
    client = openai.OpenAI(api_key=openai_api_key)

    # Use provided script_blocks if available, otherwise create our own
    if script_blocks is not None:
        blocks = script_blocks
        print(f"🎨 Using provided script blocks: {len(blocks)} blocks")
    else:
        blocks = split_script_into_blocks_improved(script_text)
        print(f"🎨 Created own script blocks: {len(blocks)} blocks")

    # Ensure we have the right number of blocks
    if len(blocks) != num_scenes:
        print(f"⚠️ Block count mismatch: {len(blocks)} blocks vs {num_scenes} expected")
        if len(blocks) > num_scenes:
            blocks = blocks[:num_scenes]
            print(f"✂️ Trimmed to {num_scenes} blocks")
        elif len(blocks) < num_scenes:
            while len(blocks) < num_scenes:
                blocks.append(blocks[-1])
            print(f"📋 Expanded to {num_scenes} blocks")

    prompts = []
    used_visuals = set()
    for i, block in enumerate(blocks):
        shot_type = SHOT_TYPES[i % len(SHOT_TYPES)]
        # Step 1: Use LLM to extract/generate a unique visual metaphor/subject/action
        system_msg = (
            "You are a world-class YouTube creative director and visual storyteller. "
            "Given a narration block, describe a unique, cinematic visual metaphor, subject, or action that best represents it. "
            "Be specific, visual, and avoid repetition across scenes. "
            "Return only the visual description (no extra text)."
        )
        user_msg = f"""
        Script Block:
        {block}
        {f'Topic: {topic}' if topic else ''}
        {f'Tone: {tone}' if tone else ''}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=80,
                temperature=0.7,
                user="unique_visual_extractor"
            )
            visual_desc = response.choices[0].message.content.strip().strip('"') if response.choices[0].message.content else ""
            # Avoid duplicate visuals
            if visual_desc in used_visuals:
                visual_desc += f" ({shot_type})"
            used_visuals.add(visual_desc)
        except Exception as e:
            print(f"⚠️ LLM visual extraction failed for block {i+1}: {e}")
            visual_desc = f"A metaphorical scene for: {block[:80]}"
        # Step 2: Build the final prompt
        style = VISUAL_STYLES[i % len(VISUAL_STYLES)]
        prompt = (
            f"{style}, {shot_type}, {visual_desc}, cinematic, visually stunning, high detail, no text, no watermark, 16:9"
        )
        prompts.append(prompt[:250])
    print(f"✅ Generated {len(prompts)} unique, metaphor-rich, script-aligned prompts")
    return prompts

def get_style_for_scene(scene_idx):
    return VISUAL_STYLES[scene_idx % len(VISUAL_STYLES)]

def get_dynamic_negative_prompt(scene_desc):
    negatives = [
        "text", "watermark", "signature", "border", "frame", "lowres", "bad anatomy", "distorted face", "extra limbs",
        "cartoon", "anime", "jpeg artifacts", "blurry", "unprofessional"
    ]
    if not any(word in scene_desc.lower() for word in ["meditate", "yoga", "spiritual"]):
        negatives += ["meditating", "sitting cross-legged", "spiritual pose", "yoga"]
    return ", ".join(negatives)

def generate_ab_prompts(scene, script_text, previous_prompts, style, negative_prompt, openai_api_key, n=2):
    client = openai.OpenAI(api_key=openai_api_key)
    ab_prompts = []
    for _ in range(n):
        system_msg = (
            "You are an expert prompt engineer for AI image generation. "
            "Given a scene description, write a unique, metaphor-rich, cinematic prompt for a YouTube video. "
            "Avoid repeated or irrelevant poses unless the scene calls for it. "
            "Always end with: 'no text, no watermark, cinematic, 16:9'."
        )
        user_msg = (
            f"Script context:\n{script_text}\n\n"
            f"Scene description:\n{scene}\n\n"
            f"Previous prompts: {previous_prompts}\n\n"
            f"Style: {style}\n"
            f"Negative prompt: {negative_prompt}\n"
            "Write a single, vivid, metaphorical, and story-aligned prompt for this scene. Max 180 characters."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=100,
            temperature=0.7,
            user="ab_prompt_generator"
        )
        prompt_raw = response.choices[0].message.content.strip().strip("\"") if response.choices[0].message.content else ""
        ab_prompts.append(prompt_raw[:250])
    return ab_prompts

def generate_scene_plan_and_prompts(script_text, openai_api_key, target_duration_minutes=5.0, ab_options=2, feedbacks=None):
    # Calculate dynamic max_scenes based on target duration (1-2 scenes per minute for optimal pacing)
    # For videos 1-3 minutes: ~2 scenes per minute
    # For videos 3-10 minutes: ~1.5 scenes per minute  
    # For videos 10+ minutes: ~1 scene per minute
    if target_duration_minutes <= 3:
        scenes_per_minute = 2.0
    elif target_duration_minutes <= 10:
        scenes_per_minute = 1.5
    else:
        scenes_per_minute = 1.0
    
    max_scenes = max(3, int(target_duration_minutes * scenes_per_minute))
    print(f"🎬 Dynamic scene calculation: {target_duration_minutes} minutes = {max_scenes} scenes ({scenes_per_minute} scenes/min)")
    
    client = openai.OpenAI(api_key=openai_api_key)
    # Step 1: Ask LLM to split script into scenes/visual moments
    system_msg = (
        "You are a world-class YouTube creative director. "
        "Given a narration script, split it into a sequence of distinct, visual scenes or moments. "
        "For each scene, describe the main action, metaphor, or object in 1-2 lines. "
        "Do NOT repeat the same pose or visual idea. Focus on diversity and story progression."
    )
    if feedbacks:
        system_msg += f" Previous feedback from last video: {feedbacks}"
    user_msg = f"Script:\n{script_text}\n\nSplit this script into {max_scenes} visual scenes. List each as: Scene X: ..."
    scene_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        max_tokens=600,
        temperature=0.5,
        user="scene_planner"
    )
    scene_content = scene_resp.choices[0].message.content
    if scene_content is None:
        scene_content = ""
    scenes = [l.split(":",1)[-1].strip() for l in scene_content.strip().split("\n") if isinstance(l, str) and l.strip() and l.lower().startswith("scene")]
    scenes = scenes[:max_scenes]
    # Step 2: For each scene, generate A/B prompt options with style rotation
    all_prompt_options = []
    for i, scene in enumerate(scenes):
        style = VISUAL_STYLES[i % len(VISUAL_STYLES)]
        ab_prompts = []
        for ab in range(ab_options):
            ab_system = (
                "You are a master visual prompt engineer for AI image generation. "
                "Given a scene description, create a unique, metaphor-rich, cinematic prompt for a YouTube video. "
                "Do NOT repeat poses or visuals from other scenes. "
                "Style: " + style + ". "
                "No text, watermark, signature, border, frame, lowres, bad anatomy, jpeg artifacts, blurry, unprofessional. "
            )
            if feedbacks:
                ab_system += f" Previous feedback: {feedbacks}"
            ab_user = f"Scene: {scene}\nWrite a prompt for this scene."
            ab_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": ab_system}, {"role": "user", "content": ab_user}],
                max_tokens=100,
                temperature=0.7+0.1*ab,
                user="ab_prompt_gen"
            )
            ab_content = ab_resp.choices[0].message.content
            if ab_content is None:
                ab_content = ""
            ab_prompt = ab_content.strip().strip('"')
            ab_prompts.append(f"{style}, {ab_prompt}")
        all_prompt_options.append(ab_prompts)
    # Step 3: LLM-based selection of best prompt for each scene
    selected_prompts = []
    for i, (scene, prompt_options) in enumerate(zip(scenes, all_prompt_options)):
        sel_system = (
            "You are an expert YouTube creative director. "
            "Given several prompt options for a scene, select the one that is most visually unique, emotionally impactful, and best aligned with the script. "
            "Do NOT select a prompt that repeats visuals from other scenes."
        )
        if feedbacks:
            sel_system += f" Previous feedback: {feedbacks}"
        sel_user = f"Scene: {scene}\nPrompt Options:\n" + "\n".join([f"Option {j+1}: {p}" for j,p in enumerate(prompt_options)]) + "\nSelect the best option and return ONLY the prompt text."
        sel_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sel_system}, {"role": "user", "content": sel_user}],
            max_tokens=80,
            temperature=0.3,
            user="prompt_selector"
        )
        sel_content = sel_resp.choices[0].message.content
        if sel_content is None:
            sel_content = ""
        best_prompt = sel_content.strip().strip('"')
        selected_prompts.append(best_prompt)
    return scenes, all_prompt_options, selected_prompts

# Utility to parse feedback files for iterative improvement
def parse_feedback_files(video_folder, num_scenes):
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

def generate_simple_visual_prompts(script_blocks, openai_api_key):
    """
    Generate simple, direct visual prompts for each script block using OpenAI LLM.
    Returns a list of prompts (one per block).
    """
    import openai
    prompts = []
    client = openai.OpenAI(api_key=openai_api_key)
    SYSTEM_MSG = (
        "You are an expert at writing short, direct prompts for AI image generation. "
        "Given a script block, describe the scene visually in one line. "
        "Be literal and clear, avoid metaphors. Always end with: 'no text, no watermark, 16:9'."
    )
    for block in script_blocks:
        user_msg = f"Script Block: {block}\nPrompt:"
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=60,
                temperature=0.4,
                user="simple_visual_prompt"
            )
            content = response.choices[0].message.content
            prompt_raw = content.strip().strip('"') if content else ""
            if "no text" not in prompt_raw.lower():
                prompt_raw = prompt_raw.rstrip(".") + ", no text, no watermark, 16:9"
            prompts.append(prompt_raw[:180])
        except Exception as e:
            print(f"⚠️ Simple visual prompt generation failed for block: {block[:40]}... Error: {e}")
            prompts.append(f"A simple illustration of: {block[:120]}, no text, no watermark, 16:9")
    return prompts

IMAGE_STYLE_LOCK = "3D Pixar movie style, ultra-realistic, expressive lighting, detailed textures, cinematic, not flat, not 2D, not comic, not illustration, no cartoon outline, no anime"

def create_viral_image_prompt(block_text: str, tone: str, topic: str, env: dict, scene_number: int = 1) -> str:
    import openai
    system_msg = (
        "You are a viral content strategist and cinematographer. "
        "Create prompts for YouTube thumbnails and scenes that maximize clicks and engagement. "
        "Focus on: cinematic realism, emotional impact, professional quality, broad appeal. "
        "Use 'elderly Indian farmer Kishan, white beard, wise eyes, traditional simple clothes' consistently. "
        "Target 12-15% CTR with viral sharing potential."
    )
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
        final_prompt = f"{IMAGE_STYLE_LOCK}, {prompt_raw}"
        print(f"[PromptGen] Viral prompt generated: {final_prompt}")
        return final_prompt[:250]
    except Exception as e:
        print(f"⚠️ Viral prompt generation failed: {e}")
        safe_block = block_text.replace("\n", " ")
        fallback = f"{IMAGE_STYLE_LOCK}, elderly Indian farmer Kishan, {safe_block[:80]}, {shot_style}, viral potential, click-worthy, no text, no watermark, 16:9 cinematic"
        print(f"[PromptGen] Fallback viral prompt: {fallback}")
        return fallback

def create_image_prompt(block_text: str, tone: str, topic: str, env: dict) -> str:
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
    final_prompt = f"{IMAGE_STYLE_LOCK}, {prompt_raw}"
    print(f"[PromptGen] Scene prompt generated: {final_prompt}")
    return final_prompt[:200]

def create_advanced_prompt(block_text: str, tone: str, topic: str, env: dict, scene_number: int = 1, target_audience: str = "general_audience") -> str:
    import openai
    import random
    style_base = IMAGE_STYLE_LOCK
    shot_type = SHOT_TYPES[(scene_number - 1) % len(SHOT_TYPES)]
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
        final_prompt = f"{IMAGE_STYLE_LOCK}, {prompt_raw}"
        print(f"[PromptGen] Advanced prompt generated: {final_prompt}")
        return final_prompt[:250]
    except Exception as e:
        print(f"⚠️ Advanced prompt generation failed: {e}")
        safe_block = block_text.replace("\n", " ")
        fallback = f"{IMAGE_STYLE_LOCK}, elderly Indian farmer Kishan, {safe_block[:80]}, {shot_type}, no text, no watermark, 16:9 cinematic"
        print(f"[PromptGen] Fallback advanced prompt: {fallback}")
        return fallback

def generate_dynamic_thumbnail_prompt(topic, script, tone):
    """
    Generate a dynamic, content-aware thumbnail prompt for AI image generation.
    Uses GPT to create a unique, visually descriptive Pixar-style scene for each video.
    Falls back to the old logic if OpenAI is unavailable or fails.
    """
    try:
        import openai
        from utils import load_env
        env = load_env()
        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise Exception("No OpenAI API key found")
        prompt = (
            f"Suggest a visually striking, click-worthy YouTube thumbnail scene for a video titled '{topic}'. "
            f"Describe a unique Pixar-style scene (character, background, action, emotion, setting, camera angle) that fits the topic and tone '{tone}'. "
            f"Do NOT include any text or watermark in the image. "
            f"Script excerpt: {script[:200]}"
        )
        response = openai.OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative visual designer for YouTube thumbnails."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.85,
            user="thumbnail_prompt_generator"
        )
        visual_scene = response.choices[0].message.content
        if visual_scene is None:
            raise Exception("GPT did not return a visual scene description")
        visual_scene = visual_scene.strip()
        # Add style/negative prompt
        IMAGE_STYLE_LOCK = "3D Pixar movie style, ultra-realistic, expressive lighting, detailed textures, cinematic, not flat, not 2D, not comic, not illustration, no cartoon outline, no anime"
        negative = (
            "no text, no watermark, 16:9, no extra hands, no deformed anatomy, "
            "no duplicate faces, no blur, no distortion, no cartoon outline, no anime artifacts, "
            "no busy background, no multiple objects, no excessive colors, no crowd, no clutter"
        )
        final_prompt = f"{visual_scene}, {IMAGE_STYLE_LOCK}, {negative}"
        print(f"[PromptGen] Thumbnail prompt: {final_prompt}")
        return final_prompt[:300]
    except Exception as e:
        print(f"⚠️ GPT thumbnail prompt failed: {e}\n   Falling back to static rules.")
        # Fallback: old logic
        IMAGE_STYLE_LOCK = "3D Pixar movie style, ultra-realistic, expressive lighting, detailed textures, cinematic, not flat, not 2D, not comic, not illustration, no cartoon outline, no anime"
        topic_lower = topic.lower()
        if "finance" in topic_lower:
            main_subject = "confident young businesswoman"
            emotion = "excited"
            action = "holding a stack of money"
        elif "motivation" in topic_lower:
            main_subject = "motivational speaker"
            emotion = "inspired"
            action = "giving a speech"
        elif "public speaking" in topic_lower:
            main_subject = "charismatic speaker"
            emotion = "confident"
            action = "speaking on stage"
        elif "student" in topic_lower:
            main_subject = "focused student"
            emotion = "determined"
            action = "studying at desk"
        else:
            main_subject = "charismatic person"
            emotion = "happy"
            action = "smiling at the camera"
        style = IMAGE_STYLE_LOCK + ", clean background, minimal color palette, single subject, professional lighting, high clarity"
        negative = (
            "no text, no watermark, 16:9, no extra hands, no deformed anatomy, "
            "no duplicate faces, no blur, no distortion, no cartoon outline, no anime artifacts, "
            "no busy background, no multiple objects, no excessive colors, no crowd, no clutter"
        )
        prompt = (
            f"Close-up of {main_subject}, {emotion} expression, {action}, {style}, "
            f"{negative}"
        )
        return prompt 