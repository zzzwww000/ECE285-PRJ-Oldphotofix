"""
utils/scheduler.py

Scheduler utilities for DACM-based image restoration.
"""
def severity_to_teff(severity_score, min_steps=20, max_steps=150):
    """
    Map severity score in [0, 1] to DDIM effective steps (Teff).
    Formula aligned with project interface:
        Teff = round(min_steps + severity * (max_steps - min_steps))
    """
    severity = float(severity_score)
    severity = max(0.0, min(1.0, severity))

    teff = round(min_steps + severity * (max_steps - min_steps))
    teff = max(min_steps, min(max_steps, int(teff)))
    return teff
