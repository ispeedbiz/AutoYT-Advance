#!/usr/bin/env python3
"""
LLM Video Generation Module
Uses LLMs to generate video scripts, scene breakdowns, and editing instructions.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import openai

class LLMVideoGenerator:
    """
    Generates video content using LLMs for script creation, scene planning, and editing instructions.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the LLM video generator.
        
        Args:
            openai_api_key: OpenAI API key for LLM access
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
    
    def generate_video_script(self, topic: str, tone: str, duration: str, 
                            audience: str, highlights: str = "", notes: str = "") -> Dict:
        """
        Generate a complete video script with scene breakdowns.
        
        Args:
            topic: Main topic of the video
            tone: Tone/style of the video
            duration: Target duration
            audience: Target audience
            highlights: Key points to include
            notes: Additional notes
            
        Returns:
            Dictionary containing script and scene breakdown
        """
        try:
            prompt = f"""
            Create a complete video script for a YouTube video with the following details:
            
            Topic: {topic}
            Tone: {tone}
            Duration: {duration}
            Target Audience: {audience}
            Key Highlights: {highlights}
            Additional Notes: {notes}
            
            Please provide:
            1. A compelling opening hook (30 seconds)
            2. Main content sections with timing
            3. Scene descriptions for each section
            4. Visual suggestions for each scene
            5. Transition ideas between scenes
            6. A strong call-to-action ending
            
            Format the response as JSON with the following structure:
            {{
                "title": "Video title",
                "hook": "Opening hook text",
                "sections": [
                    {{
                        "title": "Section title",
                        "duration": "estimated seconds",
                        "script": "narration text",
                        "visual_description": "what to show",
                        "scene_type": "close-up/wide/medium",
                        "transitions": "how to transition to next scene"
                    }}
                ],
                "ending": "Call-to-action text",
                "total_duration": "estimated total seconds"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert video script writer and director."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No response from LLM")
            
            # Parse JSON response
            script_data = json.loads(content)
            self.logger.info(f"Generated video script for topic: {topic}")
            return script_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate video script: {e}")
            return {}
    
    def generate_scene_breakdown(self, script_section: str, visual_style: str = "cinematic") -> Dict:
        """
        Generate detailed scene breakdown for a script section.
        
        Args:
            script_section: Text content for the scene
            visual_style: Visual style preference
            
        Returns:
            Dictionary with scene details
        """
        try:
            prompt = f"""
            Create a detailed scene breakdown for this script section:
            
            Script: "{script_section}"
            Visual Style: {visual_style}
            
            Provide:
            1. Camera angles and movements
            2. Lighting setup
            3. Color palette
            4. Props or elements to include
            5. Timing for each shot
            6. Emotional tone to convey
            
            Format as JSON:
            {{
                "camera_angles": ["list of camera angles"],
                "lighting": "lighting description",
                "color_palette": "color scheme",
                "props": ["list of props/elements"],
                "shot_timing": "timing breakdown",
                "emotional_tone": "emotional description",
                "visual_prompt": "detailed visual prompt for AI image generation"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a cinematographer and visual director."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.6
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No response from LLM")
            
            scene_data = json.loads(content)
            return scene_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate scene breakdown: {e}")
            return {}
    
    def generate_editing_instructions(self, video_script: Dict) -> Dict:
        """
        Generate editing instructions based on the video script.
        
        Args:
            video_script: Complete video script dictionary
            
        Returns:
            Dictionary with editing instructions
        """
        try:
            sections = video_script.get("sections", [])
            editing_plan = {
                "overall_pacing": "slow/medium/fast",
                "transition_style": "fade/cut/dissolve",
                "color_grading": "warm/cool/neutral",
                "audio_mixing": "narration_volume, music_volume, effects",
                "section_edits": []
            }
            
            for i, section in enumerate(sections):
                section_edit = {
                    "section_index": i,
                    "duration": section.get("duration", "30"),
                    "transition_in": "fade_in" if i == 0 else "cross_fade",
                    "transition_out": "cross_fade" if i < len(sections) - 1 else "fade_out",
                    "visual_effects": [],
                    "audio_effects": []
                }
                
                # Generate specific editing instructions for this section
                edit_prompt = f"""
                Generate editing instructions for this video section:
                
                Script: "{section.get('script', '')}"
                Visual: "{section.get('visual_description', '')}"
                Duration: {section.get('duration', '30')} seconds
                
                Suggest:
                1. Visual effects (zoom, pan, color adjustments)
                2. Audio effects (fade, emphasis, background music type)
                3. Pacing adjustments
                4. Any special transitions
                
                Format as JSON:
                {{
                    "visual_effects": ["list of effects"],
                    "audio_effects": ["list of audio effects"],
                    "pacing": "slow/medium/fast",
                    "special_notes": "any special instructions"
                }}
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a video editor and post-production specialist."},
                        {"role": "user", "content": edit_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.5
                )
                
                content = response.choices[0].message.content
                if content:
                    try:
                        edit_instructions = json.loads(content)
                        section_edit.update(edit_instructions)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse editing instructions for section {i}")
                
                editing_plan["section_edits"].append(section_edit)
            
            return editing_plan
            
        except Exception as e:
            self.logger.error(f"Failed to generate editing instructions: {e}")
            return {}
    
    def generate_thumbnail_concept(self, video_script: Dict) -> Dict:
        """
        Generate thumbnail concept based on video content.
        
        Args:
            video_script: Complete video script dictionary
            
        Returns:
            Dictionary with thumbnail concept
        """
        try:
            title = video_script.get("title", "")
            hook = video_script.get("hook", "")
            
            prompt = f"""
            Create a compelling YouTube thumbnail concept for this video:
            
            Title: "{title}"
            Hook: "{hook}"
            
            Design a thumbnail that:
            1. Grabs attention in 3 seconds
            2. Uses high contrast and bold colors
            3. Includes text that's readable on mobile
            4. Creates curiosity without clickbait
            5. Matches the video's tone and topic
            
            Format as JSON:
            {{
                "visual_concept": "description of the main visual",
                "color_scheme": "primary colors to use",
                "text_elements": ["list of text elements"],
                "emotional_appeal": "what emotion to evoke",
                "ai_prompt": "detailed prompt for AI image generation"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a YouTube thumbnail designer and marketing specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No response from LLM")
            
            thumbnail_concept = json.loads(content)
            return thumbnail_concept
            
        except Exception as e:
            self.logger.error(f"Failed to generate thumbnail concept: {e}")
            return {}
    
    def create_complete_video_plan(self, topic: str, tone: str, duration: str, 
                                 audience: str, highlights: str = "", notes: str = "") -> Dict:
        """
        Create a complete video plan including script, scenes, editing, and thumbnail.
        
        Args:
            topic: Main topic of the video
            tone: Tone/style of the video
            duration: Target duration
            audience: Target audience
            highlights: Key points to include
            notes: Additional notes
            
        Returns:
            Complete video plan dictionary
        """
        try:
            # Generate main script
            video_script = self.generate_video_script(topic, tone, duration, audience, highlights, notes)
            
            if not video_script:
                raise Exception("Failed to generate video script")
            
            # Generate scene breakdowns for each section
            sections = video_script.get("sections", [])
            for i, section in enumerate(sections):
                scene_breakdown = self.generate_scene_breakdown(
                    section.get("script", ""),
                    visual_style="cinematic"
                )
                sections[i]["scene_breakdown"] = scene_breakdown
            
            # Generate editing instructions
            editing_plan = self.generate_editing_instructions(video_script)
            
            # Generate thumbnail concept
            thumbnail_concept = self.generate_thumbnail_concept(video_script)
            
            # Combine everything into a complete plan
            complete_plan = {
                "video_script": video_script,
                "editing_plan": editing_plan,
                "thumbnail_concept": thumbnail_concept,
                "metadata": {
                    "topic": topic,
                    "tone": tone,
                    "duration": duration,
                    "audience": audience,
                    "highlights": highlights,
                    "notes": notes,
                    "generated_at": "timestamp"
                }
            }
            
            self.logger.info(f"Complete video plan generated for topic: {topic}")
            return complete_plan
            
        except Exception as e:
            self.logger.error(f"Failed to create complete video plan: {e}")
            return {}


def generate_video_plan(topic: str, tone: str, duration: str, audience: str, 
                       highlights: str = "", notes: str = "", openai_api_key: str = "") -> Dict:
    """
    Convenience function to generate a complete video plan.
    
    Args:
        topic: Main topic of the video
        tone: Tone/style of the video
        duration: Target duration
        audience: Target audience
        highlights: Key points to include
        notes: Additional notes
        openai_api_key: OpenAI API key
        
    Returns:
        Complete video plan dictionary
    """
    generator = LLMVideoGenerator(openai_api_key)
    return generator.create_complete_video_plan(topic, tone, duration, audience, highlights, notes)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test video plan generation
    api_key = "your-openai-api-key"  # Replace with actual key
    plan = generate_video_plan(
        topic="The Power of Morning Routines",
        tone="Motivational",
        duration="5 minutes",
        audience="Young professionals",
        highlights="Productivity, mental health, success habits",
        notes="Focus on practical tips",
        openai_api_key=api_key
    )
    
    print("Video plan generated successfully!" if plan else "Failed to generate video plan") 