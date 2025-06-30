#!/usr/bin/env python3
"""
Automated Feedback Loop Module
Automatically regenerates and improves failed content based on quality feedback.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import openai
from quality_filter import QualityFilter

class AutomatedFeedbackLoop:
    """
    Implements an automated feedback loop that regenerates failed content
    and continuously improves quality until success or max attempts reached.
    """
    
    def __init__(self, openai_api_key: str, max_attempts: int = 3, 
                 quality_threshold: float = 0.7):
        """
        Initialize the automated feedback loop.
        
        Args:
            openai_api_key: OpenAI API key for LLM access
            max_attempts: Maximum number of regeneration attempts
            quality_threshold: Minimum quality score to accept
        """
        self.openai_api_key = openai_api_key
        self.max_attempts = max_attempts
        self.quality_threshold = quality_threshold
        self.quality_filter = QualityFilter(openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Track feedback history for improvement
        self.feedback_history = []
    
    def improve_image_generation(self, original_prompt: str, scene_description: str, 
                               feedback: str, attempt: int = 1) -> Tuple[str, bool]:
        """
        Improve image generation prompt based on feedback.
        
        Args:
            original_prompt: Original image generation prompt
            scene_description: Description of the scene
            feedback: Quality feedback from previous attempt
            attempt: Current attempt number
            
        Returns:
            Tuple of (improved_prompt, success_indicator)
        """
        try:
            improvement_prompt = f"""
            Improve this image generation prompt based on the feedback:
            
            Original Prompt: "{original_prompt}"
            Scene Description: "{scene_description}"
            Feedback: "{feedback}"
            Attempt: {attempt}/{self.max_attempts}
            
            Previous attempts feedback history:
            {self._format_feedback_history()}
            
            Instructions:
            1. Address the specific issues mentioned in the feedback
            2. Make the prompt more specific and detailed
            3. Adjust visual style if needed
            4. Ensure better alignment with the scene description
            5. Add any missing visual elements
            
            Return only the improved prompt, no explanations.
            """
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at improving AI image generation prompts."},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            improved_prompt = response.choices[0].message.content
            if improved_prompt is None:
                return original_prompt, False
            
            improved_prompt = improved_prompt.strip()
            self.logger.info(f"Improved prompt for attempt {attempt}")
            return improved_prompt, True
            
        except Exception as e:
            self.logger.error(f"Failed to improve image prompt: {e}")
            return original_prompt, False
    
    def improve_script_generation(self, original_script: str, topic: str, 
                                feedback: str, attempt: int = 1) -> Tuple[str, bool]:
        """
        Improve script generation based on feedback.
        
        Args:
            original_script: Original script text
            topic: Video topic
            feedback: Quality feedback from previous attempt
            attempt: Current attempt number
            
        Returns:
            Tuple of (improved_script, success_indicator)
        """
        try:
            improvement_prompt = f"""
            Improve this video script based on the feedback:
            
            Topic: "{topic}"
            Original Script: "{original_script}"
            Feedback: "{feedback}"
            Attempt: {attempt}/{self.max_attempts}
            
            Previous attempts feedback history:
            {self._format_feedback_history()}
            
            Instructions:
            1. Address the specific issues mentioned in the feedback
            2. Improve flow and coherence
            3. Make it more engaging and emotional
            4. Ensure better alignment with the topic
            5. Fix any language or structure issues
            
            Return only the improved script, no explanations.
            """
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert video script writer."},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            improved_script = response.choices[0].message.content
            if improved_script is None:
                return original_script, False
            
            improved_script = improved_script.strip()
            self.logger.info(f"Improved script for attempt {attempt}")
            return improved_script, True
            
        except Exception as e:
            self.logger.error(f"Failed to improve script: {e}")
            return original_script, False
    
    def improve_audio_generation(self, original_text: str, feedback: str, 
                               attempt: int = 1) -> Tuple[str, bool]:
        """
        Improve audio generation text based on feedback.
        
        Args:
            original_text: Original text for TTS
            feedback: Quality feedback from previous attempt
            attempt: Current attempt number
            
        Returns:
            Tuple of (improved_text, success_indicator)
        """
        try:
            improvement_prompt = f"""
            Improve this text for text-to-speech generation based on the feedback:
            
            Original Text: "{original_text}"
            Feedback: "{feedback}"
            Attempt: {attempt}/{self.max_attempts}
            
            Instructions:
            1. Address pronunciation or clarity issues
            2. Improve pacing and rhythm
            3. Make it more natural for speech
            4. Fix any language issues
            5. Ensure proper pauses and emphasis
            
            Return only the improved text, no explanations.
            """
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at optimizing text for speech synthesis."},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=500,
                temperature=0.6
            )
            
            improved_text = response.choices[0].message.content
            if improved_text is None:
                return original_text, False
            
            improved_text = improved_text.strip()
            self.logger.info(f"Improved audio text for attempt {attempt}")
            return improved_text, True
            
        except Exception as e:
            self.logger.error(f"Failed to improve audio text: {e}")
            return original_text, False
    
    def run_image_quality_loop(self, image_path: str, prompt: str, scene_description: str,
                              image_generator_func) -> Tuple[bool, str, Dict]:
        """
        Run automated feedback loop for image quality improvement.
        
        Args:
            image_path: Path to save the generated image
            prompt: Initial image generation prompt
            scene_description: Description of the scene
            image_generator_func: Function to generate images
            
        Returns:
            Tuple of (success, final_image_path, feedback_data)
        """
        current_prompt = prompt
        feedback_data = {
            "attempts": [],
            "final_quality_score": 0.0,
            "improvements_made": []
        }
        
        for attempt in range(1, self.max_attempts + 1):
            self.logger.info(f"Image generation attempt {attempt}/{self.max_attempts}")
            
            # Generate image
            success = image_generator_func(current_prompt, image_path)
            if not success:
                self.logger.error(f"Image generation failed on attempt {attempt}")
                feedback_data["attempts"].append({
                    "attempt": attempt,
                    "prompt": current_prompt,
                    "success": False,
                    "error": "Image generation failed"
                })
                continue
            
            # Quality check
            quality_pass, confidence, reasoning, recommendations = self.quality_filter.comprehensive_quality_check(
                image_path, current_prompt, scene_description, self.quality_threshold
            )
            
            attempt_data = {
                "attempt": attempt,
                "prompt": current_prompt,
                "success": True,
                "quality_pass": quality_pass,
                "confidence": confidence,
                "reasoning": reasoning,
                "recommendations": recommendations
            }
            feedback_data["attempts"].append(attempt_data)
            
            if quality_pass:
                self.logger.info(f"Image quality check passed on attempt {attempt}")
                feedback_data["final_quality_score"] = confidence
                return True, image_path, feedback_data
            
            # Store feedback for improvement
            self._add_feedback(f"Image attempt {attempt}: {reasoning}")
            
            # Improve prompt for next attempt
            if attempt < self.max_attempts:
                improved_prompt, improvement_success = self.improve_image_generation(
                    current_prompt, scene_description, reasoning, attempt
                )
                if improvement_success:
                    current_prompt = improved_prompt
                    feedback_data["improvements_made"].append(f"Attempt {attempt}: Improved prompt")
            
            # Add delay between attempts
            time.sleep(2)
        
        self.logger.warning(f"Image quality loop failed after {self.max_attempts} attempts")
        feedback_data["final_quality_score"] = feedback_data["attempts"][-1]["confidence"]
        return False, image_path, feedback_data
    
    def run_script_quality_loop(self, topic: str, tone: str, duration: str, 
                               audience: str, script_generator_func) -> Tuple[bool, str, Dict]:
        """
        Run automated feedback loop for script quality improvement.
        
        Args:
            topic: Video topic
            tone: Video tone
            duration: Target duration
            audience: Target audience
            script_generator_func: Function to generate scripts
            
        Returns:
            Tuple of (success, final_script, feedback_data)
        """
        feedback_data = {
            "attempts": [],
            "final_quality_score": 0.0,
            "improvements_made": []
        }
        
        for attempt in range(1, self.max_attempts + 1):
            self.logger.info(f"Script generation attempt {attempt}/{self.max_attempts}")
            
            # Generate script
            script = script_generator_func(topic, tone, duration, audience)
            if not script:
                self.logger.error(f"Script generation failed on attempt {attempt}")
                feedback_data["attempts"].append({
                    "attempt": attempt,
                    "success": False,
                    "error": "Script generation failed"
                })
                continue
            
            # Simple quality check (can be enhanced with more sophisticated analysis)
            quality_score = self._evaluate_script_quality(script, topic, tone)
            
            attempt_data = {
                "attempt": attempt,
                "script": script,
                "success": True,
                "quality_score": quality_score,
                "passes_threshold": quality_score >= self.quality_threshold
            }
            feedback_data["attempts"].append(attempt_data)
            
            if quality_score >= self.quality_threshold:
                self.logger.info(f"Script quality check passed on attempt {attempt}")
                feedback_data["final_quality_score"] = quality_score
                return True, script, feedback_data
            
            # Store feedback for improvement
            feedback = f"Script quality score: {quality_score:.2f} (threshold: {self.quality_threshold})"
            self._add_feedback(f"Script attempt {attempt}: {feedback}")
            
            # Improve script for next attempt
            if attempt < self.max_attempts:
                improved_script, improvement_success = self.improve_script_generation(
                    script, topic, feedback, attempt
                )
                if improvement_success:
                    script = improved_script
                    feedback_data["improvements_made"].append(f"Attempt {attempt}: Improved script")
            
            # Add delay between attempts
            time.sleep(2)
        
        self.logger.warning(f"Script quality loop failed after {self.max_attempts} attempts")
        feedback_data["final_quality_score"] = feedback_data["attempts"][-1]["quality_score"]
        return False, script, feedback_data
    
    def run_audio_quality_loop(self, text: str, audio_generator_func) -> Tuple[bool, str, Dict]:
        """
        Run automated feedback loop for audio quality improvement.
        
        Args:
            text: Text to convert to audio
            audio_generator_func: Function to generate audio
            
        Returns:
            Tuple of (success, final_audio_path, feedback_data)
        """
        current_text = text
        feedback_data = {
            "attempts": [],
            "final_quality_score": 0.0,
            "improvements_made": []
        }
        
        for attempt in range(1, self.max_attempts + 1):
            self.logger.info(f"Audio generation attempt {attempt}/{self.max_attempts}")
            
            # Generate audio
            audio_path = audio_generator_func(current_text)
            if not audio_path:
                self.logger.error(f"Audio generation failed on attempt {attempt}")
                feedback_data["attempts"].append({
                    "attempt": attempt,
                    "text": current_text,
                    "success": False,
                    "error": "Audio generation failed"
                })
                continue
            
            # Simple quality check (can be enhanced with audio analysis)
            quality_score = self._evaluate_audio_quality(audio_path, current_text)
            
            attempt_data = {
                "attempt": attempt,
                "text": current_text,
                "audio_path": audio_path,
                "success": True,
                "quality_score": quality_score,
                "passes_threshold": quality_score >= self.quality_threshold
            }
            feedback_data["attempts"].append(attempt_data)
            
            if quality_score >= self.quality_threshold:
                self.logger.info(f"Audio quality check passed on attempt {attempt}")
                feedback_data["final_quality_score"] = quality_score
                return True, audio_path, feedback_data
            
            # Store feedback for improvement
            feedback = f"Audio quality score: {quality_score:.2f} (threshold: {self.quality_threshold})"
            self._add_feedback(f"Audio attempt {attempt}: {feedback}")
            
            # Improve text for next attempt
            if attempt < self.max_attempts:
                improved_text, improvement_success = self.improve_audio_generation(
                    current_text, feedback, attempt
                )
                if improvement_success:
                    current_text = improved_text
                    feedback_data["improvements_made"].append(f"Attempt {attempt}: Improved text")
            
            # Add delay between attempts
            time.sleep(2)
        
        self.logger.warning(f"Audio quality loop failed after {self.max_attempts} attempts")
        feedback_data["final_quality_score"] = feedback_data["attempts"][-1]["quality_score"]
        return False, audio_path, feedback_data
    
    def _evaluate_script_quality(self, script: str, topic: str, tone: str) -> float:
        """
        Evaluate script quality using LLM analysis.
        
        Args:
            script: Script text to evaluate
            topic: Video topic
            tone: Video tone
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            evaluation_prompt = f"""
            Evaluate the quality of this video script:
            
            Topic: "{topic}"
            Tone: "{tone}"
            Script: "{script}"
            
            Rate the script on a scale of 0-1 based on:
            1. Relevance to topic (0.25 weight)
            2. Tone consistency (0.25 weight)
            3. Engagement and flow (0.25 weight)
            4. Clarity and coherence (0.25 weight)
            
            Return only a number between 0 and 1, no explanation.
            """
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a video script quality evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            score_text = response.choices[0].message.content
            if score_text is None:
                return 0.5
            
            try:
                score = float(score_text.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Script quality evaluation failed: {e}")
            return 0.5
    
    def _evaluate_audio_quality(self, audio_path: str, text: str) -> float:
        """
        Evaluate audio quality (simplified implementation).
        
        Args:
            audio_path: Path to audio file
            text: Original text
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Check if audio file exists and has reasonable size
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return 0.0
            
            # Simple heuristic: longer text should produce longer audio
            file_size = audio_file.stat().st_size
            text_length = len(text)
            
            if file_size < 1000:  # Too small
                return 0.3
            elif file_size > 10000000:  # Too large
                return 0.7
            else:
                return 0.8  # Reasonable size
                
        except Exception as e:
            self.logger.error(f"Audio quality evaluation failed: {e}")
            return 0.5
    
    def _add_feedback(self, feedback: str):
        """Add feedback to history for improvement."""
        self.feedback_history.append({
            "timestamp": time.time(),
            "feedback": feedback
        })
        
        # Keep only recent feedback (last 10 items)
        if len(self.feedback_history) > 10:
            self.feedback_history = self.feedback_history[-10:]
    
    def _format_feedback_history(self) -> str:
        """Format feedback history for prompt inclusion."""
        if not self.feedback_history:
            return "No previous feedback available."
        
        formatted = []
        for item in self.feedback_history[-5:]:  # Last 5 feedback items
            formatted.append(f"- {item['feedback']}")
        
        return "\n".join(formatted)
    
    def save_feedback_report(self, report_path: str, feedback_data: dict):
        """Save feedback report to JSON file with proper path handling."""
        try:
            # Convert any Path objects to strings for JSON serialization
            def convert_paths(obj):
                if hasattr(obj, '__fspath__'):  # PathLike object
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                else:
                    return obj
            
            # Clean the feedback data
            clean_data = convert_paths(feedback_data)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Feedback report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to save feedback report: {e}")


def create_feedback_loop(openai_api_key: str, max_attempts: int = 3, 
                        quality_threshold: float = 0.7) -> AutomatedFeedbackLoop:
    """
    Convenience function to create an automated feedback loop.
    
    Args:
        openai_api_key: OpenAI API key
        max_attempts: Maximum regeneration attempts
        quality_threshold: Minimum quality score
        
    Returns:
        AutomatedFeedbackLoop instance
    """
    return AutomatedFeedbackLoop(openai_api_key, max_attempts, quality_threshold)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test feedback loop
    api_key = "your-openai-api-key"  # Replace with actual key
    feedback_loop = create_feedback_loop(api_key, max_attempts=3, quality_threshold=0.7)
    
    print("Automated feedback loop created successfully!") 