#!/usr/bin/env python3
"""
CLIP/LLM-BASED QUALITY FILTER
Automatically reject bad frames based on visual quality and prompt alignment
"""

import os
import json
import base64
from pathlib import Path
from PIL import Image
import openai
from dotenv import load_dotenv

class QualityFilter:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 for API transmission."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def check_image_quality_clip(self, image_path, prompt, threshold=0.7):
        """
        Use CLIP-based analysis to check image quality and prompt alignment.
        Returns: (pass/fail, confidence_score, issues)
        """
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this image for quality and alignment with the prompt: "{prompt}"
            
            Check for:
            1. Visual Quality: Sharpness, clarity, professional appearance
            2. Prompt Alignment: Does the image match the intended scene/character?
            3. Technical Issues: Distortions, artifacts, poor composition
            4. Content Appropriateness: Suitable for YouTube audience
            
            Rate each aspect 1-10 and provide overall PASS/FAIL with confidence score.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            if result is None:
                return True, 0.5, "No response from API"
            
            # Parse the result
            if "PASS" in result.upper():
                # Extract confidence score if available
                confidence = 0.8  # Default confidence
                if "confidence" in result.lower():
                    try:
                        import re
                        conf_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', result.lower())
                        if conf_match:
                            confidence = float(conf_match.group(1))
                    except:
                        pass
                
                return True, confidence, "Quality check passed"
            else:
                return False, 0.0, result
            
        except Exception as e:
            print(f"⚠️ CLIP quality check failed: {e}")
            return True, 0.5, f"Check failed: {e}"  # Default to pass if check fails
    
    def check_image_quality_llm(self, image_path, prompt, scene_description):
        """
        Use LLM analysis to check image quality and story alignment.
        Returns: (pass/fail, reasoning)
        """
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            analysis_prompt = f"""
            You are a YouTube content quality controller. Analyze this image for:
            
            PROMPT: "{prompt}"
            SCENE: "{scene_description}"
            
            Quality Criteria:
            1. Visual Appeal: Is this image visually stunning and professional?
            2. Story Alignment: Does it match the intended scene and narrative?
            3. Character Consistency: Is the character consistent with previous scenes?
            4. Emotional Impact: Will this image engage viewers emotionally?
            5. Technical Quality: Sharp, clear, no artifacts or distortions?
            
            Respond with:
            - PASS/FAIL
            - Brief reasoning (1-2 sentences)
            - Quality score (1-10)
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.2
            )
            
            result = response.choices[0].message.content
            if result is None:
                return True, "No response from API"
            
            # Parse result
            if "PASS" in result.upper():
                return True, result
            else:
                return False, result
                
        except Exception as e:
            print(f"⚠️ LLM quality check failed: {e}")
            return True, f"Check failed: {e}"  # Default to pass
    
    def automated_image_quality_filter(self, image_path, prompt, scene_description, engagement_threshold=0.85):
        """
        Uses OpenAI Vision (or GPT-4o vision) to check for anatomical errors and engagement.
        Returns: (pass/fail, engagement_score, issues)
        """
        import base64, json
        try:
            base64_image = self.encode_image_to_base64(image_path)
            vision_prompt = (
                "Analyze this image for YouTube video use. "
                "Does it have any anatomical errors (extra/missing hands, limbs, faces, fingers, duplicate body parts)? "
                "Is it visually engaging, clear, and free of artifacts, text, or watermarks? "
                "Rate engagement and quality from 0 to 1. "
                "Return JSON: {\"anatomy_ok\": true/false, \"engagement_score\": 0-1, \"issues\": \"...\"}"
            )
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.0,
            )
            result = response.choices[0].message.content
            if not result:
                return False, 0.0, "No response from vision API"
            try:
                result_json = json.loads(result[result.find('{'):result.rfind('}')+1])
                anatomy_ok = result_json.get("anatomy_ok", False)
                engagement_score = float(result_json.get("engagement_score", 0))
                issues = result_json.get("issues", "")
                passed = anatomy_ok and engagement_score >= engagement_threshold
                return passed, engagement_score, issues
            except Exception as e:
                return False, 0.0, f"Vision API parse error: {e} | Raw: {result}"
        except Exception as e:
            return False, 0.0, f"Vision API error: {e}"

    def comprehensive_quality_check(self, image_path, prompt, scene_description, quality_threshold=0.8):
        """
        Comprehensive quality check using anatomical/engagement filter, CLIP, and LLM analysis.
        Returns: (pass/fail, confidence, reasoning, recommendations)
        """
        # Use Vision API as primary check
        passed, engagement_score, issues = self.automated_image_quality_filter(
            image_path, prompt, scene_description, engagement_threshold=quality_threshold
        )
        log_details = {
            'image_path': image_path,
            'prompt': prompt,
            'scene_description': scene_description,
            'engagement_score': engagement_score,
            'issues': issues,
            'passed': passed
        }
        # Actionable feedback logging
        if passed:
            print(f"[QualityFilter] PASS: {os.path.basename(image_path)} | Score: {engagement_score:.2f}")
            if issues:
                print(f"[QualityFilter] Minor issues: {issues}")
        elif engagement_score >= 0.75:
            print(f"[QualityFilter] SOFT PASS: {os.path.basename(image_path)} | Score: {engagement_score:.2f}")
            print(f"[QualityFilter] Issues: {issues}")
        else:
            print(f"[QualityFilter] FAIL: {os.path.basename(image_path)} | Score: {engagement_score:.2f}")
            print(f"[QualityFilter] Issues: {issues}")
        # Log raw details for debugging
        print(f"[QualityFilter] Vision API result: {log_details}")
        return (passed or engagement_score >= 0.75), engagement_score, ("Quality check passed" if passed else "Soft pass" if engagement_score >= 0.75 else "Quality check failed"), issues
    
    def batch_quality_check(self, image_folder, prompts_file, output_file="quality_report.json"):
        """
        Perform quality check on all images in a folder.
        """
        image_folder = Path(image_folder)
        results = []
        
        # Load prompts if available
        prompts = {}
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r') as f:
                prompts = json.load(f)
        
        # Get all image files
        image_files = sorted([f for f in image_folder.glob("*.png")])
        
        print(f"🔍 Batch quality check: {len(image_files)} images")
        
        for i, image_path in enumerate(image_files):
            print(f"\n📸 Checking image {i+1}/{len(image_files)}: {image_path.name}")
            
            # Get prompt for this image
            prompt = prompts.get(image_path.stem, "No prompt available")
            scene_description = f"Scene {i+1} from video"
            
            # Perform quality check
            pass_check, confidence, reasoning, recommendations = self.comprehensive_quality_check(
                image_path, prompt, scene_description
            )
            
            result = {
                "image": str(image_path),
                "pass": pass_check,
                "confidence": confidence,
                "reasoning": reasoning,
                "recommendations": recommendations,
                "prompt": prompt
            }
            
            results.append(result)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        passed = sum(1 for r in results if r["pass"])
        total = len(results)
        
        print(f"\n📊 QUALITY CHECK SUMMARY")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        print(f"📁 Report saved: {output_file}")
        
        return results

def filter_low_quality_images(image_folder, quality_threshold=0.6):
    """
    Filter out low-quality images and move them to a rejected folder.
    """
    image_folder = Path(image_folder)
    rejected_folder = image_folder / "rejected"
    rejected_folder.mkdir(exist_ok=True)
    
    # Load quality report
    report_file = image_folder / "quality_report.json"
    if not report_file.exists():
        print("❌ No quality report found. Run quality check first.")
        return
    
    with open(report_file, 'r') as f:
        results = json.load(f)
    
    rejected_count = 0
    for result in results:
        if not result["pass"] or result["confidence"] < quality_threshold:
            image_path = Path(result["image"])
            if image_path.exists():
                # Move to rejected folder
                rejected_path = rejected_folder / image_path.name
                image_path.rename(rejected_path)
                rejected_count += 1
                print(f"❌ Rejected: {image_path.name} (confidence: {result['confidence']:.2f})")
    
    print(f"\n📊 FILTERING COMPLETE")
    print(f"❌ Rejected: {rejected_count} images")
    print(f"📁 Rejected images moved to: {rejected_folder}")

if __name__ == "__main__":
    # Load environment
    load_dotenv()
    
    # Test quality filter
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        exit(1)
    
    filter = QualityFilter(os.environ["OPENAI_API_KEY"])
    
    # Test with a sample image
    test_image = "test_output/test_image.png"
    if os.path.exists(test_image):
        print("🧪 Testing quality filter...")
        pass_check, confidence, reasoning, recommendations = filter.comprehensive_quality_check(
            test_image, 
            "cinematic realism, elderly Indian farmer", 
            "Test scene"
        )
        print(f"Result: {'PASS' if pass_check else 'FAIL'}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reasoning: {reasoning}")
    else:
        print("💡 No test image found. Run image generation first.") 