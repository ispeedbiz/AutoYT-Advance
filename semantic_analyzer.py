#!/usr/bin/env python3
"""
Semantic Analysis System for Advanced Scene Detection
Analyzes content for topic changes, emotional shifts, and complexity.
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class SemanticAnalyzer:
    def __init__(self):
        # Keywords for different content types
        self.content_keywords = {
            "story": ["एक बार", "कहानी", "गांव", "शहर", "परिवार", "दोस्त", "मिला", "गया", "आया"],
            "lesson": ["सीख", "शिक्षा", "ज्ञान", "समझ", "बताना", "सिखाना", "जानना", "समझना"],
            "motivation": ["सफलता", "लक्ष्य", "सपना", "मेहनत", "धैर्य", "विश्वास", "आत्मविश्वास", "जीत"],
            "problem_solution": ["समस्या", "समाधान", "चुनौती", "मुश्किल", "आसान", "हल", "उपाय"],
            "comparison": ["लेकिन", "परंतु", "हालांकि", "जबकि", "इसके विपरीत", "दूसरी तरफ"]
        }
        
        # Emotional keywords
        self.emotional_keywords = {
            "positive": ["खुशी", "सफलता", "जीत", "आनंद", "उत्साह", "आशा", "विश्वास", "प्रेरणा"],
            "negative": ["दुख", "हार", "निराशा", "डर", "चिंता", "तनाव", "क्रोध", "भय"],
            "neutral": ["सोच", "समझ", "जानना", "देखना", "सुनना", "महसूस", "लगना"],
            "intense": ["बहुत", "अत्यंत", "पूरी तरह", "बिल्कुल", "निश्चित", "स्पष्ट"]
        }
        
        # Transition words that indicate topic changes
        self.transition_words = [
            "लेकिन", "परंतु", "हालांकि", "इसके बाद", "फिर", "अब", "आज", "कल",
            "पहले", "बाद में", "शुरू में", "अंत में", "दूसरी तरफ", "इसके विपरीत",
            "उदाहरण के लिए", "जैसे कि", "मतलब", "यानी", "इसलिए", "क्योंकि"
        ]
        
        # Complexity indicators
        self.complexity_indicators = {
            "simple": ["और", "या", "लेकिन", "कि", "जो", "क्या", "कौन"],
            "complex": ["हालांकि", "इसके बावजूद", "जबकि", "यद्यपि", "चूंकि", "जिससे", "जिसका"]
        }
    
    def analyze_semantic_structure(self, sentences: List[str]) -> Dict:
        """
        Perform comprehensive semantic analysis of the script.
        """
        analysis = {
            "content_type": self._detect_content_type(sentences),
            "topic_changes": self._detect_topic_changes(sentences),
            "emotional_arc": self._analyze_emotional_arc(sentences),
            "complexity_profile": self._analyze_complexity(sentences),
            "narrative_structure": self._detect_narrative_structure(sentences),
            "engagement_points": self._identify_engagement_points(sentences)
        }
        
        return analysis
    
    def _detect_content_type(self, sentences: List[str]) -> Dict:
        """
        Detect the primary content type and its confidence.
        """
        content_scores = defaultdict(int)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for content_type, keywords in self.content_keywords.items():
                for keyword in keywords:
                    if keyword in sentence_lower:
                        content_scores[content_type] += 1
        
        # Find the dominant content type
        if content_scores:
            dominant_type = max(content_scores.items(), key=lambda x: x[1])
            total_matches = sum(content_scores.values())
            confidence = dominant_type[1] / total_matches if total_matches > 0 else 0
            
            return {
                "primary_type": dominant_type[0],
                "confidence": confidence,
                "all_scores": dict(content_scores)
            }
        else:
            return {
                "primary_type": "mixed",
                "confidence": 0.5,
                "all_scores": {}
            }
    
    def _detect_topic_changes(self, sentences: List[str]) -> List[int]:
        """
        Detect sentences where topic changes occur.
        """
        topic_changes = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check for transition words
            has_transition = any(word in sentence_lower for word in self.transition_words)
            
            # Check for content type shifts
            if i > 0:
                prev_sentence = sentences[i-1].lower()
                current_content = self._get_sentence_content_type(sentence_lower)
                prev_content = self._get_sentence_content_type(prev_sentence)
                
                if current_content != prev_content and has_transition:
                    topic_changes.append(i)
        
        return topic_changes
    
    def _get_sentence_content_type(self, sentence: str) -> str:
        """
        Get the content type of a single sentence.
        """
        for content_type, keywords in self.content_keywords.items():
            if any(keyword in sentence for keyword in keywords):
                return content_type
        return "neutral"
    
    def _analyze_emotional_arc(self, sentences: List[str]) -> Dict:
        """
        Analyze the emotional progression throughout the script.
        """
        emotional_scores = []
        emotional_labels = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Calculate emotional scores
            positive_score = sum(1 for word in self.emotional_keywords["positive"] if word in sentence_lower)
            negative_score = sum(1 for word in self.emotional_keywords["negative"] if word in sentence_lower)
            neutral_score = sum(1 for word in self.emotional_keywords["neutral"] if word in sentence_lower)
            intense_score = sum(1 for word in self.emotional_keywords["intense"] if word in sentence_lower)
            
            # Determine emotional label
            if positive_score > negative_score:
                if intense_score > 0:
                    emotional_labels.append("excitement")
                else:
                    emotional_labels.append("positive")
            elif negative_score > positive_score:
                if intense_score > 0:
                    emotional_labels.append("tension")
                else:
                    emotional_labels.append("negative")
            else:
                emotional_labels.append("neutral")
            
            # Store scores
            emotional_scores.append({
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score,
                "intense": intense_score,
                "net_emotion": positive_score - negative_score
            })
        
        # Analyze emotional arc
        net_emotions = [score["net_emotion"] for score in emotional_scores]
        
        return {
            "emotional_labels": emotional_labels,
            "emotional_scores": emotional_scores,
            "overall_arc": self._classify_emotional_arc(net_emotions),
            "climax_point": self._find_emotional_climax(net_emotions),
            "resolution_point": self._find_resolution_point(net_emotions)
        }
    
    def _classify_emotional_arc(self, net_emotions: List[int]) -> str:
        """
        Classify the overall emotional arc of the script.
        """
        if not net_emotions:
            return "neutral"
        
        # Calculate arc characteristics
        start_emotion = net_emotions[0]
        end_emotion = net_emotions[-1]
        max_emotion = max(net_emotions)
        min_emotion = min(net_emotions)
        
        # Classify arc
        if start_emotion < 0 and end_emotion > 0:
            return "redemption"  # Negative to positive
        elif start_emotion > 0 and end_emotion < 0:
            return "tragedy"  # Positive to negative
        elif max_emotion > 2 and min_emotion < -1:
            return "rollercoaster"  # High highs and low lows
        elif all(emotion >= 0 for emotion in net_emotions):
            return "uplifting"  # Consistently positive
        elif all(emotion <= 0 for emotion in net_emotions):
            return "melancholic"  # Consistently negative
        else:
            return "balanced"  # Mixed emotions
    
    def _find_emotional_climax(self, net_emotions: List[int]) -> int:
        """
        Find the point of maximum emotional intensity.
        """
        if not net_emotions:
            return 0
        
        # Find the point with maximum absolute emotion
        max_abs_emotion = max(abs(emotion) for emotion in net_emotions)
        for i, emotion in enumerate(net_emotions):
            if abs(emotion) == max_abs_emotion:
                return i
        
        return 0
    
    def _find_resolution_point(self, net_emotions: List[int]) -> int:
        """
        Find the point where emotional resolution occurs.
        """
        if not net_emotions:
            return 0
        
        # Look for the last significant positive emotion
        for i in range(len(net_emotions) - 1, -1, -1):
            if net_emotions[i] > 0:
                return i
        
        return len(net_emotions) - 1
    
    def _analyze_complexity(self, sentences: List[str]) -> Dict:
        """
        Analyze the complexity profile of the script.
        """
        complexity_scores = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Count complex indicators
            complex_count = sum(1 for word in self.complexity_indicators["complex"] if word in sentence_lower)
            simple_count = sum(1 for word in self.complexity_indicators["simple"] if word in sentence_lower)
            
            # Calculate complexity score
            word_count = len(sentence.split())
            complexity_score = (complex_count / max(word_count, 1)) * 100
            
            complexity_scores.append({
                "complexity_score": complexity_score,
                "complex_indicators": complex_count,
                "simple_indicators": simple_count,
                "word_count": word_count,
                "complexity_level": "high" if complexity_score > 15 else "medium" if complexity_score > 5 else "low"
            })
        
        # Overall complexity analysis
        avg_complexity = np.mean([score["complexity_score"] for score in complexity_scores])
        
        return {
            "sentence_complexity": complexity_scores,
            "average_complexity": avg_complexity,
            "complexity_trend": self._analyze_complexity_trend(complexity_scores),
            "overall_level": "high" if avg_complexity > 15 else "medium" if avg_complexity > 5 else "low"
        }
    
    def _analyze_complexity_trend(self, complexity_scores: List[Dict]) -> str:
        """
        Analyze how complexity changes throughout the script.
        """
        if len(complexity_scores) < 3:
            return "stable"
        
        scores = [score["complexity_score"] for score in complexity_scores]
        
        # Calculate trend
        first_third = np.mean(scores[:len(scores)//3])
        last_third = np.mean(scores[-len(scores)//3:])
        
        if last_third > first_third * 1.2:
            return "increasing"
        elif first_third > last_third * 1.2:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_narrative_structure(self, sentences: List[str]) -> Dict:
        """
        Detect the narrative structure of the script.
        """
        if len(sentences) < 3:
            return {"structure": "simple", "parts": []}
        
        # Look for narrative structure indicators
        structure_indicators = {
            "introduction": ["आज", "मैं", "बताना", "सिखाना", "जानना"],
            "development": ["एक बार", "कहानी", "शुरू", "पहले", "जब"],
            "climax": ["फिर", "अचानक", "लेकिन", "परंतु", "मुश्किल"],
            "resolution": ["अंत में", "सीख", "समझ", "जीत", "सफलता"]
        }
        
        structure_parts = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            for part, indicators in structure_indicators.items():
                if any(indicator in sentence_lower for indicator in indicators):
                    structure_parts.append({
                        "index": i,
                        "part": part,
                        "confidence": 0.8
                    })
                    break
        
        # Classify overall structure
        if len(structure_parts) >= 3:
            structure = "complete"  # Has introduction, development, resolution
        elif len(structure_parts) >= 2:
            structure = "partial"
        else:
            structure = "simple"
        
        return {
            "structure": structure,
            "parts": structure_parts
        }
    
    def _identify_engagement_points(self, sentences: List[str]) -> List[int]:
        """
        Identify sentences that could serve as engagement hooks.
        """
        engagement_points = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Engagement indicators
            engagement_indicators = [
                "क्या आप जानते हैं", "सोचिए", "कल्पना कीजिए", "एक सवाल",
                "रोचक बात", "आश्चर्य", "अद्भुत", "अविश्वसनीय",
                "सच्चाई", "रहस्य", "गुप्त", "खास"
            ]
            
            # Check for engagement indicators
            has_engagement = any(indicator in sentence_lower for indicator in engagement_indicators)
            
            # Check for emotional intensity
            emotional_intensity = sum(1 for word in self.emotional_keywords["intense"] if word in sentence_lower)
            
            # Check for questions
            has_question = any(char in sentence for char in ["?", "?"])
            
            if has_engagement or emotional_intensity > 1 or has_question:
                engagement_points.append(i)
        
        return engagement_points

class ContentOptimizer:
    """
    Optimizes content based on semantic analysis results.
    """
    
    def __init__(self):
        self.optimization_rules = {
            "story": {
                "scene_duration": {"min": 12, "max": 20, "optimal": 15},
                "pacing": "moderate",
                "engagement_frequency": "high"
            },
            "lesson": {
                "scene_duration": {"min": 8, "max": 15, "optimal": 10},
                "pacing": "fast",
                "engagement_frequency": "medium"
            },
            "motivation": {
                "scene_duration": {"min": 10, "max": 18, "optimal": 12},
                "pacing": "moderate",
                "engagement_frequency": "high"
            }
        }
    
    def optimize_based_on_analysis(self, scenes: List[Dict], semantic_analysis: Dict) -> List[Dict]:
        """
        Optimize scenes based on semantic analysis.
        """
        content_type = semantic_analysis.get("content_type", {}).get("primary_type", "mixed")
        emotional_arc = semantic_analysis.get("emotional_arc", {})
        complexity_profile = semantic_analysis.get("complexity_profile", {})
        
        # Apply content type optimizations
        scenes = self._apply_content_type_optimizations(scenes, content_type)
        
        # Apply emotional arc optimizations
        scenes = self._apply_emotional_optimizations(scenes, emotional_arc)
        
        # Apply complexity optimizations
        scenes = self._apply_complexity_optimizations(scenes, complexity_profile)
        
        return scenes
    
    def _apply_content_type_optimizations(self, scenes: List[Dict], content_type: str) -> List[Dict]:
        """
        Apply optimizations based on content type.
        """
        rules = self.optimization_rules.get(content_type, self.optimization_rules["lesson"])
        
        for scene in scenes:
            # Adjust scene duration based on content type
            optimal_duration = rules["scene_duration"]["optimal"]
            current_duration = scene.get("estimated_duration", 15)
            
            # Adjust if significantly different
            if abs(current_duration - optimal_duration) > 5:
                scene["estimated_duration"] = optimal_duration
                scene["metadata"]["content_type_optimized"] = True
            
            # Add pacing information
            scene["metadata"]["recommended_pacing"] = rules["pacing"]
        
        return scenes
    
    def _apply_emotional_optimizations(self, scenes: List[Dict], emotional_arc: Dict) -> List[Dict]:
        """
        Apply optimizations based on emotional arc.
        """
        climax_point = emotional_arc.get("climax_point", 0)
        resolution_point = emotional_arc.get("resolution_point", len(scenes) - 1)
        
        for i, scene in enumerate(scenes):
            # Optimize climax scene
            if i == climax_point:
                scene["estimated_duration"] *= 1.2  # Longer for impact
                scene["metadata"]["climax_optimized"] = True
            
            # Optimize resolution scene
            elif i == resolution_point:
                scene["estimated_duration"] *= 1.1  # Slightly longer for closure
                scene["metadata"]["resolution_optimized"] = True
        
        return scenes
    
    def _apply_complexity_optimizations(self, scenes: List[Dict], complexity_profile: Dict) -> List[Dict]:
        """
        Apply optimizations based on complexity profile.
        """
        overall_level = complexity_profile.get("overall_level", "medium")
        
        for scene in scenes:
            # Adjust duration based on complexity
            if overall_level == "high":
                scene["estimated_duration"] *= 1.1  # Longer for complex content
            elif overall_level == "low":
                scene["estimated_duration"] *= 0.9  # Shorter for simple content
            
            scene["metadata"]["complexity_optimized"] = True
        
        return scenes 