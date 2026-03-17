import re

def rgba_to_hex(color):
    """Convert rgba or rgb string to hex, or return hex if already hex."""
    if isinstance(color, str):
        color = color.strip()
        # If already hex
        if color.startswith('#') and (len(color) == 7 or len(color) == 4):
            return color
        # If rgba or rgb - FIX: Remove extra backslashes
        match = re.match(r"rgba?\(([^)]+)\)", color)
        if match:
            parts = match.group(1).split(',')
            r = int(float(parts[0]))
            g = int(float(parts[1]))
            b = int(float(parts[2]))
            return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    # fallback
    return '#5e2784' 