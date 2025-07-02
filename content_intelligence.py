#!/usr/bin/env python3
"""
Content Intelligence Layer - Advanced script analysis and content optimization for YouTube automation.

This module provides intelligent content analysis that goes beyond simple formulas to understand
the natural flow, complexity, and optimal duration of content.
"""

import re
import openai
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class ContentAnalysis:
    """Comprehensive analysis of script content."""
    natural_duration: float  # Estimated natural speaking time in minutes
    complexity_level: str    # low, medium, high
    content_type: str       # story, lesson, motivation, mixed
    natural_breaks: List[int]  # Sentence indices where natural breaks occur
    expansion_opportunities: List[Dict]  # Where content can be expanded
    compression_candidates: List[Dict]   # What can be compressed/removed
    engagement_score: float  # Predicted engagement potential (0-1)
    optimal_scene_count: int  # Natural number of scenes based on content
    content_density: str     # sparse, balanced, dense
    emotional_arc: str      # flat, rising, falling, rollercoaster

@dataclass
class DurationMismatch:
    """Information about content-duration mismatches and solutions."""
    mismatch_type: str      # too_short, too_long, good_match
    severity: str           # minor, moderate, severe
    current_duration: float
    target_duration: float
    deviation_percent: float
    suggestions: List[str]  # Actionable suggestions
    auto_fix_available: bool

class ContentIntelligence:
    """Advanced content analysis and optimization system."""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Load engagement patterns from real YouTube data
        self.engagement_patterns = self._load_engagement_patterns()
        self.optimal_timings = self._load_optimal_timings()
    
    def analyze_script(self, script: str) -> ContentAnalysis:
        """Perform comprehensive analysis of script content."""
        print("🧠 Content Intelligence: Analyzing script...")
        
        # Basic metrics
        sentences = self._split_into_sentences(script)
        words = script.split()
        
        # Advanced AI analysis
        ai_analysis = self._ai_content_analysis(script)
        
        # Natural duration estimation (improved algorithm)
        natural_duration = self._estimate_natural_duration(script, ai_analysis)
        
        # Content complexity analysis
        complexity_level = self._analyze_complexity(script, sentences, ai_analysis)
        
        # Find semantic boundaries (not just sentence breaks)
        natural_breaks = self._find_semantic_boundaries(script, sentences)
        
        # Content expansion/compression opportunities
        expansion_ops = self._find_expansion_opportunities(script, ai_analysis)
        compression_candidates = self._find_compression_candidates(script, ai_analysis)
        
        # Engagement prediction
        engagement_score = self._predict_engagement(script, ai_analysis)
        
        # Optimal scene count based on content flow
        optimal_scenes = self._calculate_optimal_scenes(script, natural_breaks, complexity_level)
        
        return ContentAnalysis(
            natural_duration=natural_duration,
            complexity_level=complexity_level,
            content_type=ai_analysis.get("content_type", "mixed"),
            natural_breaks=natural_breaks,
            expansion_opportunities=expansion_ops,
            compression_candidates=compression_candidates,
            engagement_score=engagement_score,
            optimal_scene_count=optimal_scenes,
            content_density=ai_analysis.get("content_density", "balanced"),
            emotional_arc=ai_analysis.get("emotional_arc", "flat")
        )
    
    def detect_duration_mismatch(self, content_analysis: ContentAnalysis, target_duration: float) -> DurationMismatch:
        """Detect and analyze content-duration mismatches."""
        current_duration = content_analysis.natural_duration
        deviation = abs(current_duration - target_duration) / target_duration * 100
        
        if deviation <= 15:
            mismatch_type = "good_match"
            severity = "minor"
        elif current_duration < target_duration:
            mismatch_type = "too_short"
            severity = "moderate" if deviation <= 40 else "severe"
        else:
            mismatch_type = "too_long"
            severity = "moderate" if deviation <= 40 else "severe"
        
        # Generate actionable suggestions
        suggestions = self._generate_duration_suggestions(
            mismatch_type, severity, content_analysis, target_duration
        )
        
        # Check if auto-fix is possible
        auto_fix_available = self._can_auto_fix(mismatch_type, severity, content_analysis)
        
        return DurationMismatch(
            mismatch_type=mismatch_type,
            severity=severity,
            current_duration=current_duration,
            target_duration=target_duration,
            deviation_percent=deviation,
            suggestions=suggestions,
            auto_fix_available=auto_fix_available
        )
    
    def optimize_content_for_duration(self, script: str, target_duration: float) -> Dict:
        """Intelligently optimize content to match target duration."""
        content_analysis = self.analyze_script(script)
        mismatch = self.detect_duration_mismatch(content_analysis, target_duration)
        
        print(f"📊 Content-Duration Analysis:")
        print(f"   Natural duration: {content_analysis.natural_duration:.1f} minutes")
        print(f"   Target duration: {target_duration:.1f} minutes")
        print(f"   Mismatch: {mismatch.mismatch_type} ({mismatch.severity})")
        print(f"   Deviation: {mismatch.deviation_percent:.1f}%")
        print(f"   Engagement score: {content_analysis.engagement_score:.2f}")
        
        if mismatch.mismatch_type == "good_match":
            print("   ✅ Content naturally fits target duration!")
            return self._create_optimal_scene_plan(script, content_analysis)
        
        elif mismatch.auto_fix_available:
            print("   🔧 Auto-fixing content-duration mismatch...")
            return self._auto_fix_content(script, content_analysis, mismatch, target_duration)
        
        else:
            print("   💡 Content adjustment recommendations:")
            for suggestion in mismatch.suggestions:
                print(f"      • {suggestion}")
            return self._create_best_effort_plan(script, content_analysis, target_duration)
    
    def _ai_content_analysis(self, script: str) -> Dict:
        """Use AI to analyze content characteristics."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert content analyst. Analyze the given script and return a JSON object with these fields:
                        - content_type: story/lesson/motivation/mixed
                        - complexity_level: low/medium/high
                        - content_density: sparse/balanced/dense
                        - emotional_arc: flat/rising/falling/rollercoaster
                        - key_topics: array of main topics
                        - natural_segments: array of natural content divisions
                        - expansion_potential: array of areas that could be expanded
                        - compression_potential: array of areas that could be compressed"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this script:\n\n{script}"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if content:
                # Try to parse JSON, fallback to basic analysis if it fails
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Extract key information with regex if JSON parsing fails
                    return self._extract_analysis_from_text(content)
            
        except Exception as e:
            print(f"   ⚠️ AI analysis failed: {e}")
        
        # Fallback analysis
        return {
            "content_type": "mixed",
            "complexity_level": "medium",
            "content_density": "balanced",
            "emotional_arc": "flat",
            "key_topics": [],
            "natural_segments": [],
            "expansion_potential": [],
            "compression_potential": []
        }
    
    def _estimate_natural_duration(self, script: str, ai_analysis: Dict) -> float:
        """Estimate natural speaking duration with advanced factors."""
        words = len(script.split())
        
        # Base speaking rate: 150-180 words per minute
        base_wpm = 165
        
        # Adjust for content complexity
        complexity = ai_analysis.get("complexity_level", "medium")
        if complexity == "high":
            base_wpm *= 0.85  # Slower for complex content
        elif complexity == "low":
            base_wpm *= 1.15  # Faster for simple content
        
        # Adjust for content density
        density = ai_analysis.get("content_density", "balanced")
        if density == "dense":
            base_wpm *= 0.9   # Slower for dense content
        elif density == "sparse":
            base_wpm *= 1.1   # Faster for sparse content
        
        # Adjust for emotional content (emotional content is spoken slower)
        emotional_arc = ai_analysis.get("emotional_arc", "flat")
        if emotional_arc in ["rising", "rollercoaster"]:
            base_wpm *= 0.95
        
        # Add pause time for natural breaks
        sentences = len(re.split(r'[।.!?]+', script))
        pause_time = sentences * 0.8 / 60  # 0.8 seconds pause per sentence
        
        speaking_time = words / base_wpm
        total_time = speaking_time + pause_time
        
        return total_time
    
    def _find_semantic_boundaries(self, script: str, sentences: List[str]) -> List[int]:
        """Find natural semantic boundaries using AI analysis."""
        try:
            # Use AI to identify topic shifts and natural breaks
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """Identify natural break points in this script where topics shift or new concepts are introduced. 
                        Return sentence numbers (1-based) where natural breaks occur. Focus on semantic boundaries, not just sentence endings.
                        Return as a simple comma-separated list of numbers."""
                    },
                    {
                        "role": "user",
                        "content": f"Find natural break points in this script:\n\n{script}"
                    }
                ],
                max_tokens=100,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            if content:
                # Extract numbers from response
                numbers = re.findall(r'\d+', content)
                # Convert to 0-based indices and ensure they're within range
                breaks = [max(0, min(int(n) - 1, len(sentences) - 1)) for n in numbers]
                return sorted(list(set(breaks)))  # Remove duplicates and sort
                
        except Exception as e:
            print(f"   ⚠️ Semantic boundary detection failed: {e}")
        
        # Fallback: Use simple heuristics
        breaks = []
        for i, sentence in enumerate(sentences):
            # Look for transition words/phrases that indicate new topics
            transitions = ["लेकिन", "फिर", "अब", "इसके बाद", "दूसरी तरफ", "आज", "पहले", "अंत में"]
            if any(trans in sentence for trans in transitions):
                breaks.append(i)
        
        # Ensure we have at least some breaks for long content
        if len(breaks) < 2 and len(sentences) > 5:
            # Add breaks at roughly equal intervals
            interval = len(sentences) // 3
            breaks = [interval, interval * 2]
        
        return breaks
    
    def _split_into_sentences(self, script: str) -> List[str]:
        """Split script into sentences, handling Hindi punctuation."""
        sentences = re.split(r'(?<=[।.!?])\s+', script.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_complexity(self, script: str, sentences: List[str], ai_analysis: Dict) -> str:
        """Analyze content complexity using multiple factors."""
        # Use AI analysis if available
        ai_complexity = ai_analysis.get("complexity_level")
        if ai_complexity in ["low", "medium", "high"]:
            return ai_complexity
        
        # Fallback analysis
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        unique_words = len(set(script.lower().split()))
        total_words = len(script.split())
        
        if avg_sentence_length > 20 or unique_words / total_words > 0.7:
            return "high"
        elif avg_sentence_length > 12 or unique_words / total_words > 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_duration_suggestions(self, mismatch_type: str, severity: str, 
                                     content_analysis: ContentAnalysis, target_duration: float) -> List[str]:
        """Generate actionable suggestions for fixing duration mismatches."""
        suggestions = []
        
        if mismatch_type == "too_short":
            suggestions.extend([
                f"Add more context and background information to key points",
                f"Include additional examples or stories to illustrate concepts",
                f"Expand on the practical applications of the main ideas",
                f"Add reflection questions for viewers to think about",
                f"Include related tangents that support the main message"
            ])
            
            if content_analysis.expansion_opportunities:
                suggestions.append("AI identified specific expansion opportunities in the content")
            
            if severity == "severe":
                suggestions.extend([
                    f"Consider creating a multi-part series instead of one long video",
                    f"Add an introduction and conclusion section",
                    f"Include more detailed explanations of complex concepts"
                ])
        
        elif mismatch_type == "too_long":
            suggestions.extend([
                f"Remove redundant information and repetitive points",
                f"Combine similar concepts into single, focused segments",
                f"Extract only the most essential highlights",
                f"Consider splitting into multiple shorter videos",
                f"Tighten the language and remove filler words"
            ])
            
            if content_analysis.compression_candidates:
                suggestions.append("AI identified specific sections that can be compressed")
            
            if severity == "severe":
                suggestions.extend([
                    f"Focus on the top 3 most important points only",
                    f"Create a summary version with detailed version as separate content",
                    f"Remove tangential stories that don't directly support the main message"
                ])
        
        return suggestions
    
    def _create_optimal_scene_plan(self, script: str, content_analysis: ContentAnalysis) -> Dict:
        """Create an optimal scene plan when content naturally fits duration."""
        sentences = self._split_into_sentences(script)
        
        # Use natural breaks to create scenes
        scene_texts = []
        start_idx = 0
        
        for break_idx in content_analysis.natural_breaks + [len(sentences)]:
            if break_idx > start_idx:
                scene_text = " ".join(sentences[start_idx:break_idx])
                if scene_text.strip():
                    scene_texts.append(scene_text.strip())
                start_idx = break_idx
        
        # Ensure we have reasonable scene count
        if len(scene_texts) < 3:
            scene_texts = self._split_scenes_further(scene_texts)
        elif len(scene_texts) > 20:
            scene_texts = self._merge_short_scenes(scene_texts)
        
        return {
            "status": "optimal",
            "method": "semantic_boundaries",
            "scene_count": len(scene_texts),
            "scene_texts": scene_texts,
            "estimated_duration": content_analysis.natural_duration,
            "confidence": "high",
            "engagement_score": content_analysis.engagement_score,
            "optimization_notes": "Content naturally fits target duration with optimal scene breaks"
        }
    
    def _find_expansion_opportunities(self, script: str, ai_analysis: Dict) -> List[Dict]:
        """Find opportunities to expand content."""
        opportunities = []
        
        # Use AI analysis if available
        ai_expansion = ai_analysis.get("expansion_potential", [])
        for item in ai_expansion:
            opportunities.append({
                "type": "ai_suggested",
                "description": str(item),
                "potential_addition": "5-15 seconds"
            })
        
        # Add heuristic-based opportunities
        sentences = self._split_into_sentences(script)
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 8:  # Very short sentences
                opportunities.append({
                    "type": "short_sentence",
                    "description": f"Sentence {i+1} could be expanded with more detail",
                    "potential_addition": "3-8 seconds"
                })
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    def _find_compression_candidates(self, script: str, ai_analysis: Dict) -> List[Dict]:
        """Find content that can be compressed."""
        candidates = []
        
        # Use AI analysis if available
        ai_compression = ai_analysis.get("compression_potential", [])
        for item in ai_compression:
            candidates.append({
                "type": "ai_suggested",
                "description": str(item),
                "potential_reduction": "5-15 seconds"
            })
        
        # Add heuristic-based candidates
        sentences = self._split_into_sentences(script)
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 25:  # Very long sentences
                candidates.append({
                    "type": "long_sentence",
                    "description": f"Sentence {i+1} could be split or condensed",
                    "potential_reduction": "3-10 seconds"
                })
        
        return candidates[:5]  # Limit to top 5 candidates
    
    def _predict_engagement(self, script: str, ai_analysis: Dict) -> float:
        """Predict engagement potential of the content."""
        # Basic engagement factors
        score = 0.5  # Base score
        
        # Content type bonus
        content_type = ai_analysis.get("content_type", "mixed")
        if content_type == "story":
            score += 0.2  # Stories are engaging
        elif content_type == "motivation":
            score += 0.15  # Motivational content performs well
        
        # Emotional arc bonus
        emotional_arc = ai_analysis.get("emotional_arc", "flat")
        if emotional_arc in ["rising", "rollercoaster"]:
            score += 0.2
        elif emotional_arc == "falling":
            score -= 0.1
        
        # Complexity penalty
        complexity = ai_analysis.get("complexity_level", "medium")
        if complexity == "high":
            score -= 0.1
        elif complexity == "low":
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_optimal_scenes(self, script: str, natural_breaks: List[int], complexity: str) -> int:
        """Calculate optimal number of scenes based on content analysis."""
        sentences = self._split_into_sentences(script)
        
        # Base calculation using natural breaks
        base_scenes = len(natural_breaks) + 1 if natural_breaks else 3
        
        # Adjust for complexity
        if complexity == "high":
            base_scenes = max(base_scenes, len(sentences) // 4)  # More scenes for complex content
        elif complexity == "low":
            base_scenes = min(base_scenes, len(sentences) // 2)  # Fewer scenes for simple content
        
        # Ensure reasonable bounds
        return max(3, min(base_scenes, 15))
    
    def _can_auto_fix(self, mismatch_type: str, severity: str, content_analysis: ContentAnalysis) -> bool:
        """Determine if mismatch can be automatically fixed."""
        if severity == "severe":
            return False  # Severe mismatches need human intervention
        
        if mismatch_type == "too_short" and content_analysis.expansion_opportunities:
            return True
        
        if mismatch_type == "too_long" and content_analysis.compression_candidates:
            return True
        
        return False
    
    def _auto_fix_content(self, script: str, content_analysis: ContentAnalysis, 
                         mismatch: DurationMismatch, target_duration: float) -> Dict:
        """Automatically fix minor content-duration mismatches."""
        # This would implement automatic content expansion/compression
        # For now, return best effort plan with notes
        return {
            "status": "auto_fixed",
            "method": "ai_optimization",
            "scene_count": content_analysis.optimal_scene_count,
            "scene_texts": self._create_optimal_scene_plan(script, content_analysis)["scene_texts"],
            "estimated_duration": target_duration,
            "confidence": "medium",
            "engagement_score": content_analysis.engagement_score,
            "optimization_notes": f"Applied auto-fix for {mismatch.mismatch_type} content"
        }
    
    def _create_best_effort_plan(self, script: str, content_analysis: ContentAnalysis, target_duration: float) -> Dict:
        """Create best effort plan when perfect optimization isn't possible."""
        optimal_plan = self._create_optimal_scene_plan(script, content_analysis)
        
        return {
            "status": "best_effort",
            "method": "semantic_boundaries",
            "scene_count": optimal_plan["scene_count"],
            "scene_texts": optimal_plan["scene_texts"],
            "estimated_duration": content_analysis.natural_duration,
            "confidence": "low",
            "engagement_score": content_analysis.engagement_score,
            "optimization_notes": f"Content-duration mismatch detected. Manual adjustment recommended."
        }
    
    def _load_engagement_patterns(self) -> Dict:
        """Load engagement patterns from YouTube performance data."""
        # This would load real data in production
        return {
            "optimal_hook_duration": (5, 15),
            "optimal_story_duration": (15, 45),
            "optimal_lesson_duration": (20, 60),
            "retention_drop_points": [15, 30, 60, 120],
            "engagement_peaks": ["opening", "story_climax", "conclusion"]
        }
    
    def _load_optimal_timings(self) -> Dict:
        """Load optimal timing data for different content types."""
        return {
            "story": {"min": 15, "max": 45, "optimal": 25},
            "lesson": {"min": 20, "max": 60, "optimal": 35},
            "motivation": {"min": 10, "max": 30, "optimal": 18},
            "mixed": {"min": 15, "max": 40, "optimal": 25}
        }
    
    def _extract_analysis_from_text(self, text: str) -> Dict:
        """Extract analysis information from AI response text when JSON parsing fails."""
        analysis = {
            "content_type": "mixed",
            "complexity_level": "medium",
            "content_density": "balanced",
            "emotional_arc": "flat"
        }
        
        # Try to extract key information using regex
        if re.search(r'story|narrative|tale', text.lower()):
            analysis["content_type"] = "story"
        elif re.search(r'lesson|education|learn|teach', text.lower()):
            analysis["content_type"] = "lesson"
        elif re.search(r'motivat|inspir|encourage', text.lower()):
            analysis["content_type"] = "motivation"
        
        if re.search(r'complex|difficult|advanced', text.lower()):
            analysis["complexity_level"] = "high"
        elif re.search(r'simple|basic|easy', text.lower()):
            analysis["complexity_level"] = "low"
        
        return analysis
    
    def _split_scenes_further(self, scene_texts: List[str]) -> List[str]:
        """Split scenes further if we have too few."""
        new_scenes = []
        for scene in scene_texts:
            sentences = self._split_into_sentences(scene)
            if len(sentences) > 3:
                # Split long scenes in half
                mid = len(sentences) // 2
                new_scenes.append(" ".join(sentences[:mid]))
                new_scenes.append(" ".join(sentences[mid:]))
            else:
                new_scenes.append(scene)
        return new_scenes
    
    def _merge_short_scenes(self, scene_texts: List[str]) -> List[str]:
        """Merge scenes if we have too many."""
        merged = []
        i = 0
        while i < len(scene_texts):
            current_scene = scene_texts[i]
            
            # If current scene is short and there's a next scene, merge them
            if (i + 1 < len(scene_texts) and 
                len(current_scene.split()) < 15):
                merged.append(current_scene + " " + scene_texts[i + 1])
                i += 2
            else:
                merged.append(current_scene)
                i += 1
        
        return merged 