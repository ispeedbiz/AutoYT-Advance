import requests
import json
import time

API_URL = "http://127.0.0.1:8188"
WORKFLOW_FILE = "workflow/prompt_ultra_quality.json"

# pylint: disable=bare-except
def generate_image(prompt_text, out_file="result_image.png", negative_prompt=None, max_wait_sec=900, quality="ultra"):
    """
    Generate image with specified quality level.
    
    Args:
        prompt_text: The prompt for image generation
        out_file: Output file path
        negative_prompt: Negative prompt (optional)
        max_wait_sec: Maximum wait time in seconds
        quality: Quality level - "standard", "advanced", or "ultra" (default)
    """
    import os
    from pathlib import Path
    
    # Select workflow based on quality
    quality_workflows = {
        "standard": "workflow/prompt.json",
        "advanced": "workflow/prompt_advanced.json", 
        "ultra": "workflow/prompt_ultra_quality.json"
    }
    
    workflow_file = quality_workflows.get(quality, "workflow/prompt_ultra_quality.json")
    print(f"🎨 Using {quality.upper()} quality workflow: {workflow_file}")
    
    with open(workflow_file, "r") as f:
        workflow = json.load(f)

    # First CLIPTextEncode with non-empty prompt is positive
    positive_found = False
    for node in workflow.values():
        if node.get("class_type") == "CLIPTextEncode":
            if not positive_found and node["inputs"].get("text", "").strip():
                node["inputs"]["text"] = prompt_text
                positive_found = True
            elif negative_prompt and positive_found:
                # treat next CLIPTextEncode as negative
                node["inputs"]["text"] = negative_prompt
                break

    print("📤 Sending prompt to ComfyUI...")
    res = requests.post(f"{API_URL}/prompt", json={"prompt": workflow})
    res.raise_for_status()
    prompt_id = res.json()["prompt_id"]
    
    print(f"🔄 Waiting for image generation (prompt_id: {prompt_id})...")

    output_filename = None
    waited = 0
    check_interval = 3  # Check every 3 seconds
    
    while waited < max_wait_sec:
        time.sleep(check_interval)
        waited += check_interval
        
        try:
            # Check if generation is complete
            hist_response = requests.get(f"{API_URL}/history/{prompt_id}")
            if hist_response.status_code != 200:
                continue
                
            hist = hist_response.json()
            
            if prompt_id in hist:
                prompt_history = hist[prompt_id]
                
                # Check if generation is complete
                if "outputs" in prompt_history:
                    for node_id, node_output in prompt_history["outputs"].items():
                        if "images" in node_output and node_output["images"]:
                            output_filename = node_output["images"][0]["filename"]
                            break
                    
                    if output_filename:
                        print(f"✅ Image generated: {output_filename}")
                        break
                        
                # Check for errors
                elif "status" in prompt_history and prompt_history["status"].get("completed", False):
                    print("⚠️ Generation completed but no images found")
                    break
                    
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            continue
            
        # Show progress
        if waited % 15 == 0:  # Every 15 seconds
            print(f"🔄 Still generating... ({waited}s/{max_wait_sec}s)")

    if not output_filename:
        print(f"❌ Image generation timed out after {waited}s")
        return False

    # Download the generated image
    try:
        print(f"📥 Downloading image: {output_filename}")
        img_response = requests.get(f"{API_URL}/view?filename={output_filename}&type=output")
        
        if img_response.status_code == 200:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            with open(out_file, "wb") as f:
                f.write(img_response.content)
            
            # Verify file was created and has content
            if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:  # At least 1KB
                print(f"✅ Image saved successfully: {out_file}")
                print(f"📊 File size: {os.path.getsize(out_file) // 1024}KB")
                return True
            else:
                print(f"❌ Image file is too small or corrupted")
                return False
        else:
            print(f"❌ Failed to download image: HTTP {img_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return False

def batch_generate(prompts, folder="generated_images"):
    import os
    os.makedirs(folder, exist_ok=True)
    
    for idx, prompt in enumerate(prompts):
        out_file = os.path.join(folder, f"scene_{idx+1:02}.png")
        success = generate_image(prompt, out_file)
        if not success:
            print(f"⚠️ Generation failed for prompt: {prompt}")

if __name__ == "__main__":
    test_prompts = ["A wise owl reading, Ghibli style", "An astronaut meditating on Mars, Ghibli style"]
    batch_generate(test_prompts)
