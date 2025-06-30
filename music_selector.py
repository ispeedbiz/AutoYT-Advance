#!/usr/bin/env python3
"""
Dynamic Background Music Selection System
Selects appropriate background music based on content analysis.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional
from semantic_analyzer import SemanticAnalyzer

class MusicSelector:
    """
    Selects background music based on content analysis and emotional tone.
    """
    
    def __init__(self, backgrounds_dir: Path):
        self.backgrounds_dir = backgrounds_dir
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Music file characteristics (based on typical instrumental music analysis)
        self.music_characteristics = {
            "Pensive Piano - Audionautix.mp3": {
                "mood": "contemplative",
                "energy": "low",
                "tempo": "slow",
                "emotions": ["reflection", "calm", "melancholic", "neutral"],
                "content_types": ["story", "lesson", "motivation"],
                "intensity": 0.3
            },
            "Serenity - Aakash Gandhi.mp3": {
                "mood": "peaceful",
                "energy": "low",
                "tempo": "slow",
                "emotions": ["calm", "positive", "reflection", "uplifting"],
                "content_types": ["motivation", "lesson", "story"],
                "intensity": 0.4
            },
            "Renunciation - Asher Fulero.mp3": {
                "mood": "dramatic",
                "energy": "medium",
                "tempo": "moderate",
                "emotions": ["tension", "excitement", "redemption", "balanced"],
                "content_types": ["story", "motivation"],
                "intensity": 0.7
            },
            "T'as où les vaches_ - Dan Bodan.mp3": {
                "mood": "playful",
                "energy": "medium",
                "tempo": "moderate",
                "emotions": ["positive", "excitement", "uplifting", "balanced"],
                "content_types": ["motivation", "story"],
                "intensity": 0.6
            },
            "Dreamland - Aakash Gandhi.mp3": {
                "mood": "dreamy",
                "energy": "low",
                "tempo": "slow",
                "emotions": ["calm", "reflection", "neutral", "melancholic"],
                "content_types": ["story", "lesson"],
                "intensity": 0.3
            },
            "Allégro - Emmit Fenn.mp3": {
                "mood": "energetic",
                "energy": "high",
                "tempo": "fast",
                "emotions": ["excitement", "positive", "uplifting", "tension"],
                "content_types": ["motivation", "story"],
                "intensity": 0.8
            }
        }
    
    def select_music_for_content(self, script: str) -> str:
        """
        Select the most appropriate background music based on script analysis.
        
        Args:
            script: The script text to analyze
            
        Returns:
            Filename of the selected music track
        """
        try:
            # Analyze the script content
            sentences = self._split_into_sentences(script)
            if not sentences:
                return self._get_random_music()
            
            analysis = self.semantic_analyzer.analyze_semantic_structure(sentences)
            
            # Extract key characteristics
            content_type = analysis.get("content_type", {}).get("primary_type", "mixed")
            emotional_arc = analysis.get("emotional_arc", {}).get("overall_arc", "balanced")
            complexity = analysis.get("complexity_profile", {}).get("overall_level", "medium")
            
            # Select music based on analysis
            selected_music = self._select_based_on_analysis(content_type, emotional_arc, complexity)
            
            print(f"🎵 Music Selection Analysis:")
            print(f"   Content Type: {content_type}")
            print(f"   Emotional Arc: {emotional_arc}")
            print(f"   Complexity: {complexity}")
            print(f"   Selected Music: {selected_music}")
            
            return selected_music
            
        except Exception as e:
            print(f"⚠️ Music selection failed: {e}")
            return self._get_random_music()
    
    def _split_into_sentences(self, script: str) -> List[str]:
        """Split script into sentences for analysis."""
        import re
        sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _select_based_on_analysis(self, content_type: str, emotional_arc: str, complexity: str) -> str:
        """
        Select music based on content analysis.
        """
        # Score each music track based on how well it matches the content
        music_scores = {}
        
        for filename, characteristics in self.music_characteristics.items():
            score = 0
            
            # Content type matching
            if content_type in characteristics["content_types"]:
                score += 3
            elif content_type == "mixed":
                score += 1
            
            # Emotional arc matching
            if emotional_arc in characteristics["emotions"]:
                score += 4
            elif emotional_arc == "balanced":
                score += 2
            
            # Complexity matching
            if complexity == "high" and characteristics["intensity"] > 0.6:
                score += 2
            elif complexity == "low" and characteristics["intensity"] < 0.5:
                score += 2
            elif complexity == "medium":
                score += 1
            
            # Energy level matching based on emotional arc
            if emotional_arc in ["excitement", "tension", "uplifting"] and characteristics["energy"] in ["medium", "high"]:
                score += 2
            elif emotional_arc in ["calm", "reflection", "melancholic"] and characteristics["energy"] == "low":
                score += 2
            
            music_scores[filename] = score
        
        # Select the music with the highest score
        if music_scores:
            best_music = max(music_scores.items(), key=lambda x: x[1])
            if best_music[1] > 0:  # If we have a good match
                return best_music[0]
        
        # Fallback to random selection
        return self._get_random_music()
    
    def _get_random_music(self) -> str:
        """Get a random music file from available tracks."""
        available_music = list(self.music_characteristics.keys())
        if available_music:
            selected = random.choice(available_music)
            print(f"🎵 Random music selected: {selected}")
            return selected
        else:
            # Fallback to a default
            return "Pensive Piano - Audionautix.mp3"
    
    def get_music_path(self, filename: str) -> Optional[Path]:
        """Get the full path to a music file."""
        music_path = self.backgrounds_dir / filename
        if music_path.exists():
            return music_path
        else:
            print(f"⚠️ Music file not found: {music_path}")
            return None
    
    def get_music_metadata(self, filename: str) -> Dict:
        """Get metadata for a music file."""
        return self.music_characteristics.get(filename, {})

class DynamicAudioProcessor:
    """
    Handles dynamic audio processing including volume adjustment based on content.
    """
    
    def __init__(self):
        self.volume_adjustments = {
            "excitement": -12,  # Lower background for exciting content
            "tension": -10,     # Lower background for tense content
            "calm": -14,        # Higher background for calm content
            "reflection": -13,  # Moderate background for reflection
            "neutral": -14,     # Standard background level
            "uplifting": -11,   # Moderate background for uplifting content
            "melancholic": -14  # Higher background for melancholic content
        }
    
    def get_volume_adjustment(self, emotional_arc: str) -> int:
        """Get volume adjustment in dB based on emotional content."""
        return self.volume_adjustments.get(emotional_arc, -14)
    
    def get_fade_duration(self, scene_duration: float) -> float:
        """Calculate appropriate fade duration based on scene length."""
        if scene_duration < 10:
            return 1.0  # Short fade for short scenes
        elif scene_duration < 20:
            return 2.0  # Medium fade for medium scenes
        else:
            return 3.0  # Longer fade for long scenes 