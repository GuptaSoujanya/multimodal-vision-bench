import re

def relaxed_chart_match(pred: str, target: str, tolerance: float = 0.05) -> bool:
    """
    Standard relaxed accuracy used in ChartQA benchmarks (within 5% tolerance for numbers).
    """
    clean_p = pred.replace("%", "").replace("$", "").replace(",", "").strip()
    clean_t = target.replace("%", "").replace("$", "").replace(",", "").strip()

    num_p = re.findall(r"[-+]?\d*\.?\d+", clean_p)
    num_t = re.findall(r"[-+]?\d*\.?\d+", clean_t)

    if num_p and num_t:
        try:
            val_p = float(num_p[0])
            val_t = float(num_t[0])
            if abs(val_t) < 1e-6:
                return abs(val_p - val_t) < 1e-4
            return abs(val_p - val_t) / abs(val_t) <= tolerance
        except ValueError:
            pass

    return clean_p.lower() == clean_t.lower()
