"""
Advanced Analytics Utilities for Industrial Maintenance Platform
Provides health scoring, risk classification, and diagnostic analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def calculate_health_score(
    efficiency_index: float,
    vibration_index: float,
    thermal_index: float
) -> float:
    """
    Calculate composite health score (0-100) using weighted indices
    
    Weighting strategy:
    - Efficiency Index: 50% (most direct indicator of overall health)
    - Vibration Index: 30% (critical for mechanical health)
    - Thermal Index: 20% (important but often secondary)
    
    Args:
        efficiency_index: Efficiency percentage (0-100, higher is better)
        vibration_index: Vibration score (0-100, lower is better)
        thermal_index: Thermal score (0-100, lower is better)
    
    Returns:
        Health score (0-100, higher is better)
    """
    # Normalize efficiency (already 0-100, higher is better)
    efficiency_component = efficiency_index
    
    # Invert vibration and thermal (convert "lower is better" to "higher is better")
    vibration_component = 100 - vibration_index
    thermal_component = 100 - thermal_index
    
    # Weighted average
    health_score = (
        0.50 * efficiency_component +
        0.30 * vibration_component +
        0.20 * thermal_component
    )
    
    # Clamp to 0-100
    return max(0, min(100, health_score))


def classify_risk_level(health_score: float) -> str:
    """
    Classify risk level based on health score
    
    Args:
        health_score: Health score (0-100)
    
    Returns:
        Risk level: "Low", "Medium", "High", or "Critical"
    """
    if health_score >= 80:
        return "Low"
    elif health_score >= 60:
        return "Medium"
    elif health_score >= 40:
        return "High"
    else:
        return "Critical"


def identify_dominant_issue(
    efficiency_index: float,
    vibration_index: float,
    thermal_index: float
) -> str:
    """
    Identify the dominant issue affecting machine health
    
    Logic:
    - If efficiency is significantly degraded but indices are normal -> Efficiency Issue
    - If vibration is high -> Vibration (Mechanical) Issue
    - If thermal is high -> Thermal (Overheating) Issue
    - If multiple issues -> Combined Issue
    - If all indices are good -> Operational
    
    Args:
        efficiency_index: Efficiency percentage (0-100, higher is better)
        vibration_index: Vibration score (0-100, lower is better)
        thermal_index: Thermal score (0-100, lower is better)
    
    Returns:
        Issue type: "Vibration", "Thermal", "Efficiency", "Combined", or "Operational"
    """
    # Define thresholds
    vibration_threshold = 60
    thermal_threshold = 60
    efficiency_threshold = 70
    
    # Check issues
    vibration_issue = vibration_index > vibration_threshold
    thermal_issue = thermal_index > thermal_threshold
    efficiency_issue = efficiency_index < efficiency_threshold
    
    # Count issues
    issue_count = sum([vibration_issue, thermal_issue, efficiency_issue])
    
    if issue_count == 0:
        return "Operational"
    elif issue_count >= 2:
        return "Combined"
    elif vibration_issue:
        return "Vibration"
    elif thermal_issue:
        return "Thermal"
    else:
        return "Efficiency"


def enrich_predictions_with_analytics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich prediction dataframe with advanced analytics
    
    Adds:
    - health_score: Composite health metric (0-100)
    - risk_level: Risk classification (Low/Medium/High/Critical)
    - dominant_issue: Primary issue type
    
    Args:
        predictions_df: DataFrame with columns: efficiency_index, vibration_index, thermal_index
    
    Returns:
        Enriched DataFrame with additional analytics columns
    """
    df = predictions_df.copy()
    
    # Calculate health scores
    df['health_score'] = df.apply(
        lambda row: calculate_health_score(
            row['efficiency_index'],
            row['vibration_index'],
            row['thermal_index']
        ),
        axis=1
    )
    
    # Classify risk levels
    df['risk_level'] = df['health_score'].apply(classify_risk_level)
    
    # Identify dominant issues
    df['dominant_issue'] = df.apply(
        lambda row: identify_dominant_issue(
            row['efficiency_index'],
            row['vibration_index'],
            row['thermal_index']
        ),
        axis=1
    )
    
    return df


def calculate_fleet_statistics(predictions_df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate fleet-wide statistics for dashboard metrics
    
    Args:
        predictions_df: DataFrame with prediction data
    
    Returns:
        Dictionary with fleet statistics
    """
    enriched_df = enrich_predictions_with_analytics(predictions_df)
    
    return {
        'total_assets': len(enriched_df),
        'avg_health_score': enriched_df['health_score'].mean(),
        'avg_efficiency': enriched_df['efficiency_index'].mean(),
        'avg_vibration': enriched_df['vibration_index'].mean(),
        'avg_thermal': enriched_df['thermal_index'].mean(),
        'risk_distribution': enriched_df['risk_level'].value_counts().to_dict(),
        'issue_distribution': enriched_df['dominant_issue'].value_counts().to_dict(),
        'critical_count': len(enriched_df[enriched_df['risk_level'] == 'Critical']),
        'high_risk_count': len(enriched_df[enriched_df['risk_level'] == 'High']),
        'medium_risk_count': len(enriched_df[enriched_df['risk_level'] == 'Medium']),
        'low_risk_count': len(enriched_df[enriched_df['risk_level'] == 'Low'])
    }


def get_machine_analytics(
    machine_index: int,
    uploaded_data: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Get comprehensive analytics for a specific machine
    
    Args:
        machine_index: Index of the machine in the dataframe
        uploaded_data: Original uploaded sensor data
        predictions_df: Predictions dataframe
    
    Returns:
        Tuple of (machine_data_dict, prediction_data_dict)
    """
    # Get machine sensor data
    machine_row = uploaded_data.iloc[machine_index]
    machine_data = machine_row.to_dict()
    
    # Get prediction data
    prediction_row = predictions_df.iloc[machine_index]
    prediction_data = {
        'vibration_index': prediction_row['vibration_index'],
        'thermal_index': prediction_row['thermal_index'],
        'efficiency_index': prediction_row['efficiency_index']
    }
    
    # Add enriched analytics
    enriched_predictions = enrich_predictions_with_analytics(predictions_df)
    enriched_row = enriched_predictions.iloc[machine_index]
    
    prediction_data['health_score'] = enriched_row['health_score']
    prediction_data['risk_level'] = enriched_row['risk_level']
    prediction_data['dominant_issue'] = enriched_row['dominant_issue']
    
    return machine_data, prediction_data