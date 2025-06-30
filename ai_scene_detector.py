#!/usr/bin/env python3
"""
AI-Powered Scene Detection System
Uses GPT to intelligently identify natural scene breaks based on content analysis.
"""

import openai
import json
from typing import List, Dict, Tuple
import re

class AISceneDetector:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def detect_scene_breaks(self, script: str, target_duration_minutes: float) -> List[Dict]:
        """
        Use AI to detect natural scene breaks in the script.
        
        Args:
            script: The full script text
            target_duration_minutes: Target video duration
            
        Returns:
            List of scene objects with metadata
        """
        try:
            # Split into sentences first
            sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                return self._create_simple_scenes(sentences)
            
            # Use AI to analyze content and suggest breaks
            scene_analysis = self._analyze_content_with_ai(sentences, target_duration_minutes)
            
            # Create scenes based on AI analysis
            scenes = self._create_scenes_from_analysis(sentences, scene_analysis)
            
            return scenes
            
        except Exception as e:
            print(f"⚠️ AI scene detection failed: {e}")
            return self._fallback_scene_detection(script, target_duration_minutes)
    
    def _analyze_content_with_ai(self, sentences: List[str], target_duration_minutes: float) -> Dict:
        """
        Use GPT to analyze content and suggest optimal scene breaks.
        """
        # Prepare the analysis prompt
        analysis_prompt = self._create_analysis_prompt(sentences, target_duration_minutes)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert video editor and content analyst. 
                    Analyze the given script and suggest optimal scene breaks based on:
                    1. Topic changes and thematic shifts
                    2. Emotional intensity and dramatic moments
                    3. Natural narrative flow and story structure
                    4. Optimal pacing for viewer engagement
                    5. Content type (story, lesson, motivation, etc.)
                    
                    Return a JSON object with scene break suggestions."""
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            content = response.choices[0].message.content
            if content is None:
                raise Exception("AI response content is None")
            analysis = json.loads(content)
            return analysis
        except Exception as e:
            print(f"⚠️ Failed to parse AI analysis: {e}")
            return self._create_default_analysis(sentences, target_duration_minutes)
    
    def _create_analysis_prompt(self, sentences: List[str], target_duration_minutes: float) -> str:
        """
        Create a detailed prompt for AI content analysis.
        """
        script_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
        
        return f"""
        Analyze this script for optimal scene breaks:
        
        Target Duration: {target_duration_minutes} minutes
        Total Sentences: {len(sentences)}
        
        Script:
        {script_text}
        
        Please analyze and return a JSON object with:
        {{
            "content_type": "story|lesson|motivation|mixed",
            "emotional_arc": "description of emotional progression",
            "topic_shifts": [list of sentence indices where topics change],
            "dramatic_moments": [list of sentence indices with high emotional impact],
            "suggested_scenes": [
                {{
                    "start_sentence": 0,
                    "end_sentence": 2,
                    "scene_type": "introduction|development|climax|resolution",
                    "emotional_tone": "calm|tension|excitement|reflection",
                    "key_message": "main point of this scene",
                    "estimated_duration": 15.5
                }}
            ],
            "pacing_recommendations": "fast|moderate|slow",
            "engagement_hooks": [list of sentence indices that could be hooks]
        }}
        
        Focus on creating engaging, natural scene breaks that maintain narrative flow.
        """
    
    def _create_scenes_from_analysis(self, sentences: List[str], analysis: Dict) -> List[Dict]:
        """
        Create scene objects from AI analysis.
        """
        scenes = []
        
        for scene_data in analysis.get("suggested_scenes", []):
            start_idx = scene_data.get("start_sentence", 0)
            end_idx = scene_data.get("end_sentence", len(sentences) - 1)
            
            # Ensure valid indices
            start_idx = max(0, min(start_idx, len(sentences) - 1))
            end_idx = max(start_idx, min(end_idx, len(sentences) - 1))
            
            # Extract scene text
            scene_sentences = sentences[start_idx:end_idx + 1]
            scene_text = " ".join(scene_sentences)
            
            # Calculate metrics
            word_count = len(scene_text.split())
            estimated_duration = word_count / 2.0  # 2 words per second
            
            scene = {
                "text": scene_text,
                "start_sentence": start_idx,
                "end_sentence": end_idx,
                "word_count": word_count,
                "estimated_duration": estimated_duration,
                "scene_type": scene_data.get("scene_type", "development"),
                "emotional_tone": scene_data.get("emotional_tone", "neutral"),
                "key_message": scene_data.get("key_message", ""),
                "ai_confidence": 0.9,
                "metadata": {
                    "content_type": analysis.get("content_type", "mixed"),
                    "pacing": analysis.get("pacing_recommendations", "moderate"),
                    "is_engagement_hook": start_idx in analysis.get("engagement_hooks", [])
                }
            }
            
            scenes.append(scene)
        
        return scenes
    
    def _create_default_analysis(self, sentences: List[str], target_duration_minutes: float) -> Dict:
        """
        Create a default analysis when AI fails.
        """
        target_scenes = max(3, int(target_duration_minutes * 2))  # 2 scenes per minute
        sentences_per_scene = max(1, len(sentences) // target_scenes)
        
        suggested_scenes = []
        for i in range(0, len(sentences), sentences_per_scene):
            start_sentence = i
            end_sentence = min(i + sentences_per_scene - 1, len(sentences) - 1)
            
            scene = {
                "start_sentence": start_sentence,
                "end_sentence": end_sentence,
                "scene_type": "development",
                "emotional_tone": "neutral",
                "key_message": f"Scene {len(suggested_scenes) + 1}",
                "estimated_duration": 15.0
            }
            suggested_scenes.append(scene)
        
        return {
            "content_type": "mixed",
            "emotional_arc": "linear progression",
            "topic_shifts": [],
            "dramatic_moments": [],
            "suggested_scenes": suggested_scenes,
            "pacing_recommendations": "moderate",
            "engagement_hooks": []
        }
    
    def _create_simple_scenes(self, sentences: List[str]) -> List[Dict]:
        """
        Create simple scenes for very short scripts.
        """
        scenes = []
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            estimated_duration = word_count / 2.0
            
            scene = {
                "text": sentence,
                "start_sentence": i,
                "end_sentence": i,
                "word_count": word_count,
                "estimated_duration": estimated_duration,
                "scene_type": "simple",
                "emotional_tone": "neutral",
                "key_message": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                "ai_confidence": 0.7,
                "metadata": {
                    "content_type": "simple",
                    "pacing": "fast",
                    "is_engagement_hook": False
                }
            }
            scenes.append(scene)
        
        return scenes
    
    def _fallback_scene_detection(self, script: str, target_duration_minutes: float) -> List[Dict]:
        """
        Fallback scene detection when AI analysis fails.
        """
        sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return self._create_simple_scenes(sentences)

class DynamicPacingOptimizer:
    """
    Optimizes scene duration based on content type and engagement patterns.
    """
    
    def __init__(self):
        self.content_type_durations = {
            "story": {"min": 12, "max": 20, "optimal": 15},
            "lesson": {"min": 8, "max": 15, "optimal": 10},
            "motivation": {"min": 10, "max": 18, "optimal": 12},
            "mixed": {"min": 10, "max": 20, "optimal": 15}
        }
        
        self.emotional_tone_durations = {
            "calm": {"multiplier": 1.2},  # Longer for calm content
            "tension": {"multiplier": 0.8},  # Shorter for tension
            "excitement": {"multiplier": 0.9},  # Slightly shorter for excitement
            "reflection": {"multiplier": 1.3},  # Longer for reflection
            "neutral": {"multiplier": 1.0}  # Standard duration
        }
    
    def optimize_scene_duration(self, scene: Dict) -> float:
        """
        Optimize scene duration based on content type and emotional tone.
        """
        content_type = scene.get("metadata", {}).get("content_type", "mixed")
        emotional_tone = scene.get("emotional_tone", "neutral")
        
        # Get base duration for content type
        content_duration = self.content_type_durations.get(content_type, self.content_type_durations["mixed"])
        base_duration = content_duration["optimal"]
        
        # Apply emotional tone multiplier
        tone_multiplier = self.emotional_tone_durations.get(emotional_tone, self.emotional_tone_durations["neutral"])["multiplier"]
        
        # Calculate optimized duration
        optimized_duration = base_duration * tone_multiplier
        
        # Ensure within bounds
        min_duration = content_duration["min"]
        max_duration = content_duration["max"]
        optimized_duration = max(min_duration, min(max_duration, optimized_duration))
        
        return optimized_duration
    
    def adjust_for_engagement_hooks(self, scenes: List[Dict]) -> List[Dict]:
        """
        Adjust scene timing to optimize for engagement hooks.
        """
        for scene in scenes:
            if scene.get("metadata", {}).get("is_engagement_hook", False):
                # Make engagement hooks slightly shorter for better impact
                scene["estimated_duration"] *= 0.9
                scene["metadata"]["engagement_optimized"] = True
        
        return scenes

class RetentionOptimizer:
    """
    Optimizes scene timing based on audience retention patterns.
    """
    
    def __init__(self):
        # Typical retention drop-off points (in seconds)
        self.retention_patterns = {
            "introduction": {"critical_points": [5, 15], "drop_off_risk": "high"},
            "development": {"critical_points": [10, 25], "drop_off_risk": "medium"},
            "climax": {"critical_points": [8, 20], "drop_off_risk": "low"},
            "resolution": {"critical_points": [12, 30], "drop_off_risk": "medium"}
        }
    
    def optimize_for_retention(self, scenes: List[Dict]) -> List[Dict]:
        """
        Optimize scene timing to minimize drop-off at critical points.
        """
        total_duration = 0
        
        for scene in scenes:
            scene_type = scene.get("scene_type", "development")
            pattern = self.retention_patterns.get(scene_type, self.retention_patterns["development"])
            
            # Check if scene duration hits critical retention points
            scene_duration = scene.get("estimated_duration", 15)
            critical_points = pattern["critical_points"]
            
            # Adjust duration to avoid critical drop-off points
            for critical_point in critical_points:
                if abs(scene_duration - critical_point) < 3:  # Within 3 seconds of critical point
                    if pattern["drop_off_risk"] == "high":
                        scene_duration *= 0.8  # Reduce by 20%
                    elif pattern["drop_off_risk"] == "medium":
                        scene_duration *= 0.9  # Reduce by 10%
            
            scene["estimated_duration"] = scene_duration
            scene["metadata"]["retention_optimized"] = True
            total_duration += scene_duration
        
        return scenes 