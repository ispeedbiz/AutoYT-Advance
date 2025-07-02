#!/usr/bin/env python3
"""
Engagement Optimizer - YouTube performance-based scene optimization.

This module optimizes scene timing and structure based on real YouTube engagement patterns
and retention data to maximize viewer engagement.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EngagementPattern:
    """YouTube engagement pattern data."""
    optimal_hook_duration: Tuple[int, int]      # (min, max) seconds for opening hook
    optimal_story_duration: Tuple[int, int]     # (min, max) seconds for story segments
    optimal_lesson_duration: Tuple[int, int]    # (min, max) seconds for educational content
    retention_drop_points: List[int]            # Time points where viewers typically drop off
    engagement_peaks: List[str]                 # Types of content that drive engagement

@dataclass
class SceneOptimization:
    """Optimization recommendations for a scene."""
    scene_index: int
    current_duration: float
    optimal_duration: float
    optimization_type: str      # hook, story, lesson, transition, climax, conclusion
    engagement_score: float     # Predicted engagement (0-1)
    adjustments: List[str]      # Recommended adjustments
    priority: str              # high, medium, low

class EngagementOptimizer:
    """Optimizes video scenes for maximum YouTube engagement."""
    
    def __init__(self):
        self.engagement_patterns = self._load_youtube_patterns()
        self.scene_types = self._load_scene_type_patterns()
        self.retention_patterns = self._load_retention_patterns()
    
    def optimize_scenes_for_engagement(self, scene_texts: List[str], content_analysis: Dict) -> Dict:
        """Optimize scene structure and timing for maximum engagement."""
        print("📈 Engagement Optimizer: Analyzing scenes for YouTube performance...")
        
        # Classify each scene by type and importance
        scene_classifications = self._classify_scenes(scene_texts, content_analysis)
        
        # Calculate optimal duration for each scene based on type
        scene_optimizations = []
        total_optimized_duration = 0
        
        for i, (scene_text, classification) in enumerate(zip(scene_texts, scene_classifications)):
            optimization = self._optimize_scene(i, scene_text, classification, content_analysis)
            scene_optimizations.append(optimization)
            total_optimized_duration += optimization.optimal_duration
        
        # Apply retention-based adjustments
        scene_optimizations = self._apply_retention_optimization(scene_optimizations)
        
        # Calculate overall engagement score
        overall_engagement = self._calculate_overall_engagement(scene_optimizations, content_analysis)
        
        print(f"   📊 Engagement analysis complete:")
        print(f"   🎯 Overall engagement score: {overall_engagement:.2f}")
        print(f"   ⏱️  Optimized total duration: {total_optimized_duration/60:.1f} minutes")
        print(f"   🔧 Optimization recommendations: {len([s for s in scene_optimizations if s.adjustments])} scenes")
        
        return {
            "optimized_scenes": scene_optimizations,
            "overall_engagement_score": overall_engagement,
            "total_duration": total_optimized_duration / 60,  # Convert to minutes
            "optimization_summary": self._create_optimization_summary(scene_optimizations),
            "engagement_recommendations": self._generate_engagement_recommendations(scene_optimizations)
        }
    
    def _classify_scenes(self, scene_texts: List[str], content_analysis: Dict) -> List[Dict]:
        """Classify each scene by type and engagement potential."""
        classifications = []
        
        for i, scene_text in enumerate(scene_texts):
            # Determine scene type based on position and content
            scene_type = self._determine_scene_type(i, scene_text, len(scene_texts))
            
            # Calculate base engagement score
            engagement_score = self._calculate_scene_engagement(scene_text, scene_type, content_analysis)
            
            # Determine importance/priority
            priority = self._determine_scene_priority(i, scene_type, len(scene_texts))
            
            classifications.append({
                "type": scene_type,
                "engagement_score": engagement_score,
                "priority": priority,
                "word_count": len(scene_text.split()),
                "complexity": self._assess_scene_complexity(scene_text)
            })
        
        return classifications
    
    def _determine_scene_type(self, index: int, scene_text: str, total_scenes: int) -> str:
        """Determine the type of scene based on position and content."""
        # Position-based classification
        if index == 0:
            return "hook"  # Opening scene
        elif index == total_scenes - 1:
            return "conclusion"  # Closing scene
        elif index == 1 and total_scenes > 3:
            return "context"  # Setup after hook
        
        # Content-based classification
        text_lower = scene_text.lower()
        
        # Look for story indicators
        story_indicators = ["कहानी", "story", "एक बार", "था", "किसान", "राजा", "साधु"]
        if any(indicator in text_lower for indicator in story_indicators):
            return "story"
        
        # Look for lesson/educational indicators  
        lesson_indicators = ["सीख", "lesson", "सिखाता", "समझना", "जानना", "तरीका", "method"]
        if any(indicator in text_lower for indicator in lesson_indicators):
            return "lesson"
        
        # Look for motivational indicators
        motivation_indicators = ["सफलता", "success", "प्रेरणा", "motivation", "सोचिए", "विश्वास"]
        if any(indicator in text_lower for indicator in motivation_indicators):
            return "motivation"
        
        # Look for transition indicators
        transition_indicators = ["लेकिन", "फिर", "अब", "इसके बाद", "दूसरी तरफ"]
        if any(indicator in text_lower for indicator in transition_indicators):
            return "transition"
        
        # Default to development
        return "development"
    
    def _calculate_scene_engagement(self, scene_text: str, scene_type: str, content_analysis: Dict) -> float:
        """Calculate predicted engagement score for a scene."""
        base_score = 0.5
        
        # Scene type modifiers
        type_modifiers = {
            "hook": 0.3,        # Hooks are critical for engagement
            "story": 0.25,      # Stories are highly engaging
            "motivation": 0.2,  # Motivational content performs well
            "lesson": 0.1,      # Educational content is moderately engaging
            "conclusion": 0.15, # Strong endings help retention
            "context": 0.05,    # Context is necessary but less engaging
            "transition": -0.1, # Transitions can lose attention
            "development": 0.0  # Neutral development content
        }
        
        score = base_score + type_modifiers.get(scene_type, 0.0)
        
        # Content quality modifiers
        word_count = len(scene_text.split())
        
        # Optimal word count bonus (not too short, not too long)
        if 15 <= word_count <= 35:
            score += 0.1
        elif word_count < 8 or word_count > 50:
            score -= 0.1
        
        # Emotional content bonus
        emotional_words = ["खुशी", "दुख", "प्रेम", "डर", "गुस्सा", "आश्चर्य", "success", "failure", "love", "fear"]
        if any(word in scene_text.lower() for word in emotional_words):
            score += 0.1
        
        # Question engagement bonus
        if "?" in scene_text or "सोचिए" in scene_text or "क्या" in scene_text:
            score += 0.15
        
        # Complexity penalty for overly complex content
        complexity = self._assess_scene_complexity(scene_text)
        if complexity == "high":
            score -= 0.1
        elif complexity == "low":
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _assess_scene_complexity(self, scene_text: str) -> str:
        """Assess the complexity level of scene content."""
        words = scene_text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Complex indicators
        complex_indicators = ["however", "nevertheless", "furthermore", "consequently"]
        has_complex_words = any(indicator in scene_text.lower() for indicator in complex_indicators)
        
        if avg_word_length > 7 or has_complex_words:
            return "high"
        elif avg_word_length > 5:
            return "medium"
        else:
            return "low"
    
    def _determine_scene_priority(self, index: int, scene_type: str, total_scenes: int) -> str:
        """Determine optimization priority for a scene."""
        # Critical scenes (high priority)
        if scene_type == "hook":
            return "high"  # Opening hook is critical
        elif index == 0 or index == total_scenes - 1:
            return "high"  # First and last scenes are critical
        
        # Important scenes (medium priority)  
        elif scene_type in ["story", "motivation", "conclusion"]:
            return "medium"
        
        # Supporting scenes (low priority)
        else:
            return "low"
    
    def _optimize_scene(self, index: int, scene_text: str, classification: Dict, content_analysis: Dict) -> SceneOptimization:
        """Optimize a single scene for engagement."""
        scene_type = classification["type"]
        current_duration = len(scene_text.split()) / 2.5  # Rough estimation: 2.5 words per second
        
        # Get optimal duration range for this scene type
        optimal_range = self.scene_types.get(scene_type, {"min": 8, "max": 25, "optimal": 15})
        optimal_duration = optimal_range["optimal"]
        
        # Adjust for content complexity
        if classification["complexity"] == "high":
            optimal_duration *= 1.2  # More time for complex content
        elif classification["complexity"] == "low":
            optimal_duration *= 0.9  # Less time for simple content
        
        # Adjust for priority
        if classification["priority"] == "high":
            optimal_duration = max(optimal_duration, optimal_range["min"] * 1.2)
        
        # Generate specific adjustments
        adjustments = self._generate_scene_adjustments(
            current_duration, optimal_duration, scene_type, classification
        )
        
        return SceneOptimization(
            scene_index=index,
            current_duration=current_duration,
            optimal_duration=optimal_duration,
            optimization_type=scene_type,
            engagement_score=classification["engagement_score"],
            adjustments=adjustments,
            priority=classification["priority"]
        )
    
    def _generate_scene_adjustments(self, current_duration: float, optimal_duration: float, 
                                  scene_type: str, classification: Dict) -> List[str]:
        """Generate specific adjustment recommendations for a scene."""
        adjustments = []
        duration_diff = optimal_duration - current_duration
        
        if abs(duration_diff) <= 2:  # Within 2 seconds is fine
            return adjustments
        
        if duration_diff > 2:  # Scene is too short
            if scene_type == "hook":
                adjustments.append("Add a compelling question or bold statement to grab attention")
            elif scene_type == "story":
                adjustments.append("Include more vivid details and emotional elements")
            elif scene_type == "lesson":
                adjustments.append("Add practical examples or expand explanations")
            elif scene_type == "motivation":
                adjustments.append("Include more inspiring language and call-to-action elements")
            else:
                adjustments.append(f"Expand content with more detail (add ~{duration_diff:.0f} seconds)")
        
        elif duration_diff < -2:  # Scene is too long
            if scene_type == "hook":
                adjustments.append("Shorten to get to the point faster - hooks should be punchy")
            elif scene_type == "transition":
                adjustments.append("Minimize transition time - keep it brief and smooth")
            else:
                adjustments.append(f"Condense content for better pacing (reduce ~{abs(duration_diff):.0f} seconds)")
        
        # Engagement-specific adjustments
        if classification["engagement_score"] < 0.4:
            adjustments.append("Low engagement predicted - consider adding emotional hook or question")
        
        if classification["complexity"] == "high":
            adjustments.append("Complex content detected - consider simplifying or breaking down")
        
        return adjustments
    
    def _apply_retention_optimization(self, scene_optimizations: List[SceneOptimization]) -> List[SceneOptimization]:
        """Apply retention-based optimizations to avoid drop-off points."""
        cumulative_time = 0
        
        for optimization in scene_optimizations:
            cumulative_time += optimization.optimal_duration
            
            # Check if we're approaching known drop-off points
            for drop_point in self.retention_patterns["drop_off_points"]:
                if abs(cumulative_time - drop_point) <= 5:  # Within 5 seconds of drop-off point
                    # Adjust this scene to be more engaging
                    if optimization.optimization_type != "hook":  # Don't reduce hooks
                        optimization.optimal_duration *= 0.9  # Slightly shorter to avoid drop-off
                        optimization.adjustments.append(f"Shortened to avoid {drop_point}s retention drop-off point")
                    break
        
        return scene_optimizations
    
    def _calculate_overall_engagement(self, scene_optimizations: List[SceneOptimization], content_analysis: Dict) -> float:
        """Calculate overall predicted engagement score for the video."""
        if not scene_optimizations:
            return 0.5
        
        # Weight scenes by their importance
        weighted_scores = []
        for opt in scene_optimizations:
            weight = {"high": 3.0, "medium": 2.0, "low": 1.0}[opt.priority]
            weighted_scores.append(opt.engagement_score * weight)
        
        base_score = sum(weighted_scores) / sum(3.0 if opt.priority == "high" else 2.0 if opt.priority == "medium" else 1.0 
                                               for opt in scene_optimizations)
        
        # Content type bonus
        content_type = content_analysis.get("content_type", "mixed")
        type_bonus = {"story": 0.1, "motivation": 0.08, "lesson": 0.05, "mixed": 0.0}[content_type]
        
        # Structure bonus for good pacing
        if len(scene_optimizations) >= 3:  # Has proper structure
            structure_bonus = 0.05
        else:
            structure_bonus = -0.05
        
        return max(0.0, min(1.0, base_score + type_bonus + structure_bonus))
    
    def _create_optimization_summary(self, scene_optimizations: List[SceneOptimization]) -> Dict:
        """Create a summary of optimization recommendations."""
        total_adjustments = sum(len(opt.adjustments) for opt in scene_optimizations)
        high_priority_scenes = len([opt for opt in scene_optimizations if opt.priority == "high"])
        avg_engagement = sum(opt.engagement_score for opt in scene_optimizations) / len(scene_optimizations)
        
        return {
            "total_scenes": len(scene_optimizations),
            "scenes_needing_adjustment": len([opt for opt in scene_optimizations if opt.adjustments]),
            "high_priority_scenes": high_priority_scenes,
            "average_engagement_score": avg_engagement,
            "total_adjustments": total_adjustments,
            "optimization_status": "excellent" if total_adjustments <= 2 else "good" if total_adjustments <= 5 else "needs_work"
        }
    
    def _generate_engagement_recommendations(self, scene_optimizations: List[SceneOptimization]) -> List[str]:
        """Generate overall engagement recommendations."""
        recommendations = []
        
        # Check for weak opening
        if scene_optimizations and scene_optimizations[0].engagement_score < 0.6:
            recommendations.append("⚠️ Strengthen opening hook - critical for viewer retention")
        
        # Check for weak scenes
        weak_scenes = [opt for opt in scene_optimizations if opt.engagement_score < 0.4]
        if len(weak_scenes) > len(scene_optimizations) * 0.3:  # More than 30% weak
            recommendations.append("⚠️ Multiple low-engagement scenes detected - consider restructuring")
        
        # Check for pacing issues
        long_scenes = [opt for opt in scene_optimizations if opt.optimal_duration > 45]
        if long_scenes:
            recommendations.append("⚠️ Some scenes may be too long - consider breaking down for better pacing")
        
        # Positive recommendations
        high_engagement_scenes = [opt for opt in scene_optimizations if opt.engagement_score > 0.7]
        if len(high_engagement_scenes) > len(scene_optimizations) * 0.6:  # More than 60% strong
            recommendations.append("✅ Strong engagement potential - well-structured content")
        
        if not recommendations:
            recommendations.append("✅ Good engagement optimization - minor adjustments only")
        
        return recommendations
    
    def _load_youtube_patterns(self) -> EngagementPattern:
        """Load YouTube engagement patterns from performance data."""
        return EngagementPattern(
            optimal_hook_duration=(5, 15),
            optimal_story_duration=(15, 45),
            optimal_lesson_duration=(20, 60),
            retention_drop_points=[15, 30, 60, 120, 180],  # Common drop-off points in seconds
            engagement_peaks=["opening_hook", "story_climax", "conclusion", "call_to_action"]
        )
    
    def _load_scene_type_patterns(self) -> Dict:
        """Load optimal timing patterns for different scene types."""
        return {
            "hook": {"min": 5, "max": 15, "optimal": 8},          # Quick, punchy opening
            "story": {"min": 15, "max": 45, "optimal": 25},       # Engaging narrative
            "lesson": {"min": 20, "max": 60, "optimal": 35},      # Educational content
            "motivation": {"min": 10, "max": 30, "optimal": 18},  # Inspiring content
            "conclusion": {"min": 8, "max": 20, "optimal": 12},   # Strong ending
            "context": {"min": 5, "max": 15, "optimal": 10},      # Background info
            "transition": {"min": 2, "max": 8, "optimal": 4},     # Quick transitions
            "development": {"min": 10, "max": 30, "optimal": 18}  # General content
        }
    
    def _load_retention_patterns(self) -> Dict:
        """Load viewer retention patterns from YouTube analytics."""
        return {
            "drop_off_points": [15, 30, 60, 120, 180, 300],  # Seconds where viewers commonly leave
            "engagement_windows": [(0, 15), (25, 45), (90, 120)],  # High-engagement time windows
            "critical_moments": [8, 30, 90],  # Most critical retention points
            "optimal_total_duration": (60, 300)  # Sweet spot for total video length
        } 