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

def generate_metaphor_rich_prompts(script_text, num_scenes, openai_api_key):
    import openai
    import re
    from utils import split_script_into_blocks_improved
    client = openai.OpenAI(api_key=openai_api_key)
    blocks = split_script_into_blocks_improved(script_text)
    prompts = []

    SYSTEM_MSG = """
You are an expert visual prompt engineer for AI image generation. Given a script block, write a concise, vivid, and style-appropriate prompt for a YouTube video scene. See these examples:

Script Block: एक साधु जंगल में ध्यान कर रहे थे, चारों ओर हरियाली थी।
Prompt: watercolor painting, elderly Indian sadhu meditating in a sunlit forest, silhouette, no text, no watermark, 16:9

Script Block: गाँव में सुबह का समय था, पक्षी चहचहा रहे थे।
Prompt: cinematic realism, golden hour, birds singing in a peaceful village at dawn, no text, no watermark, 16:9

Script Block: एक बच्चा नदी के किनारे खेल रहा था।
Prompt: Studio Ghibli anime style, young Indian child playing by a river, vibrant colors, no text, no watermark, 16:9

Script Block: आकाश में इंद्रधनुष था, खेत हरे-भरे थे।
Prompt: cinematic photography, rainbow in the sky over lush green fields, no text, no watermark, 16:9

Instructions:
- Identify and describe the main character or subject of this scene based on the script block.
- If there is no person, focus on the environment, mood, or metaphor.
- Use award winning cinematic, artistic, or photographic styles as appropriate.
- Avoid photorealistic faces; use artistic, silhouette, or backlit styles for people.
- Always end with: "no text, no watermark, 16:9"
"""

    for i, block in enumerate(blocks):
        user_msg = f"Script Block: {block}\nPrompt:"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=100,
            temperature=0.7,
            user="yt_visual_prompt"
        )
        content = response.choices[0].message.content
        prompt_raw = content.strip().strip("\"") if content else ""
        # Always ensure ending
        if "no text" not in prompt_raw.lower():
            prompt_raw = prompt_raw.rstrip(".") + ", no text, no watermark, 16:9"
        prompts.append(prompt_raw[:250])
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

def generate_scene_plan_and_prompts(script_text, openai_api_key, max_scenes=6, ab_options=2, feedbacks=None):
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