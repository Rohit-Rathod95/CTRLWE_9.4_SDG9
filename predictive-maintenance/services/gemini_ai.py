"""
Google Gemini AI Service for Industrial Maintenance Intelligence
Provides real AI-powered diagnostics, root cause analysis, and maintenance recommendations
"""

import os
from typing import Dict, Any, Optional
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiAIService:
    """Service class for Google Gemini AI integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI service
        
        Args:
            api_key: Google AI API key. If None, tries to get from environment
        """
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        self.model = None
        self.is_configured = False
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.is_configured = True
            except Exception as e:
                print(f"Gemini configuration error: {e}")
                self.is_configured = False
    
    def generate_maintenance_analysis(
        self,
        machine_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        analysis_depth: str = "Standard"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered maintenance analysis
        
        Args:
            machine_data: Dictionary with sensor readings (air_temperature_k, process_temperature_k, etc.)
            prediction_data: Dictionary with ML predictions (vibration_index, thermal_index, efficiency_index)
            analysis_depth: One of ["Quick Scan", "Standard", "Deep Analysis"]
        
        Returns:
            Dictionary with analysis results including:
                - root_cause: Primary failure diagnosis
                - risk_assessment: Failure risk explanation
                - maintenance_recommendations: Actionable recommendations
                - timeline: Maintenance timeline
                - cost_impact: Financial impact analysis
                - status: "success" or "error"
                - error_message: Error details if status is "error"
        """
        if not self.is_configured:
            return {
                "status": "error",
                "error_message": "Gemini AI not configured. Please set GOOGLE_AI_API_KEY environment variable."
            }
        
        try:
            # Build context-rich prompt
            prompt = self._build_analysis_prompt(machine_data, prediction_data, analysis_depth)
            
            # Generate AI response
            response = self.model.generate_content(prompt)
            
            # Parse structured response
            analysis = self._parse_ai_response(response.text, prediction_data)
            analysis["status"] = "success"
            
            return analysis
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"AI analysis failed: {str(e)}"
            }
    
    def _build_analysis_prompt(
        self,
        machine_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        analysis_depth: str
    ) -> str:
        """Build comprehensive prompt for AI analysis"""
        
        # Determine analysis scope
        depth_instructions = {
            "Quick Scan": "Provide a concise analysis focusing on the most critical issues.",
            "Standard": "Provide a balanced analysis with root cause, recommendations, and timeline.",
            "Deep Analysis": "Provide comprehensive analysis including detailed root cause analysis, failure progression, environmental factors, and long-term strategy."
        }
        
        prompt = f"""You are an expert industrial maintenance engineer analyzing equipment sensor data and ML predictions.

**Analysis Depth:** {analysis_depth}
{depth_instructions.get(analysis_depth, depth_instructions["Standard"])}

**Machine Sensor Data:**
{self._format_sensor_data(machine_data)}

**ML Prediction Outputs:**
- Vibration Index: {prediction_data.get('vibration_index', 'N/A'):.1f} (Lower is better, >60 is concerning)
- Thermal Index: {prediction_data.get('thermal_index', 'N/A'):.1f} (Lower is better, >60 is concerning)
- Efficiency Index: {prediction_data.get('efficiency_index', 'N/A'):.1f}% (Higher is better, <70% needs attention)

**Your Task:**
Provide a structured analysis with the following sections:

1. **ROOT CAUSE DIAGNOSIS** (2-3 sentences)
   - What is the primary failure mode?
   - What sensor readings support this diagnosis?

2. **RISK ASSESSMENT** (2-3 sentences)
   - What is the current failure risk level? (Low/Medium/High/Critical)
   - What is the estimated time until failure?
   - What are the consequences of inaction?

3. **MAINTENANCE RECOMMENDATIONS** (4-6 bullet points)
   - Immediate actions (0-6 hours)
   - Short-term maintenance (1-2 days)
   - Long-term preventive measures
   - Specific parts/procedures to address

4. **MAINTENANCE TIMELINE**
   - Immediate (0-6 hours): [action]
   - Short-term (1-2 days): [action]
   - Medium-term (1-2 weeks): [action]
   - Long-term (ongoing): [action]

5. **COST IMPACT ANALYSIS** (2-3 sentences)
   - Estimated monthly loss due to degradation
   - Potential catastrophic failure cost
   - ROI of preventive maintenance

Respond in a professional, data-driven tone suitable for plant managers and maintenance engineers.
"""
        return prompt
    
    def _format_sensor_data(self, machine_data: Dict[str, Any]) -> str:
        """Format sensor data for prompt"""
        lines = []
        
        # Common sensor fields
        sensor_fields = {
            'air_temperature_k': 'Air Temperature',
            'process_temperature_k': 'Process Temperature',
            'rotational_speed_rpm': 'Rotational Speed',
            'torque_nm': 'Torque',
            'tool_wear_min': 'Tool Wear',
            'temperature': 'Ambient Temperature',
            'humidity': 'Humidity',
            'rainfall': 'Rainfall'
        }
        
        for field, label in sensor_fields.items():
            if field in machine_data:
                value = machine_data[field]
                if field.endswith('_k'):
                    # Convert Kelvin to Celsius for readability
                    celsius = value - 273.15
                    lines.append(f"- {label}: {value:.1f}K ({celsius:.1f}°C)")
                elif field == 'humidity':
                    lines.append(f"- {label}: {value}%")
                elif field == 'rainfall':
                    lines.append(f"- {label}: {value}mm")
                else:
                    lines.append(f"- {label}: {value}")
        
        return "\n".join(lines) if lines else "- No sensor data available"
    
    def _parse_ai_response(self, response_text: str, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse AI response into structured format
        
        This is a simple parser that extracts sections from the response.
        In production, you might use more sophisticated NLP parsing.
        """
        # Extract sections using simple text parsing
        sections = {
            "root_cause": "",
            "risk_assessment": "",
            "maintenance_recommendations": "",
            "timeline": "",
            "cost_impact": "",
            "full_response": response_text
        }
        
        # Try to extract sections (simplified parsing)
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line_upper = line.upper()
            if 'ROOT CAUSE' in line_upper or 'DIAGNOSIS' in line_upper:
                current_section = 'root_cause'
            elif 'RISK ASSESSMENT' in line_upper or 'RISK LEVEL' in line_upper:
                current_section = 'risk_assessment'
            elif 'MAINTENANCE RECOMMENDATION' in line_upper or 'RECOMMENDATIONS' in line_upper:
                current_section = 'maintenance_recommendations'
            elif 'TIMELINE' in line_upper or 'MAINTENANCE TIMELINE' in line_upper:
                current_section = 'timeline'
            elif 'COST IMPACT' in line_upper or 'FINANCIAL IMPACT' in line_upper:
                current_section = 'cost_impact'
            elif current_section and line.strip():
                sections[current_section] += line + "\n"
        
        # Clean up sections
        for key in sections:
            if key != 'full_response':
                sections[key] = sections[key].strip()
        
        # Add prediction data for reference
        sections['prediction_data'] = prediction_data
        
        return sections


def get_ai_service(api_key: Optional[str] = None) -> GeminiAIService:
    """
    Factory function to get AI service instance
    
    Args:
        api_key: Google AI API key. If None, tries to get from environment
    
    Returns:
        GeminiAIService instance
    """
    return GeminiAIService(api_key=api_key)