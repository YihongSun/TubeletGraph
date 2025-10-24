import base64
import io
from typing import List, Tuple, Union, Optional
import numpy as np
from PIL import Image

class HTMLWriter:
    """A simple wrapper for writing HTML files with images and text."""
    
    def __init__(self, title: str = "Generated HTML"):
        """Initialize the HTML writer with a title."""
        self.title = title
        self.content = []
        self._header = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .image-container {{ margin: 15px 0; }}
        img {{ max-width: 100%; }}
        h1, h2, h3 {{ color: #333; }}
        p {{ line-height: 1.5; }}
    </style>
</head>
<body>
<h1>{title}</h1>
"""
        self._footer = """
</body>
</html>
"""

    def add_heading(self, text: str, level: int = 2):
        """Add a heading to the HTML file.
        
        Args:
            text: The heading text
            level: The heading level (1-6)
        """
        level = max(1, min(6, level))  # Ensure level is between 1 and 6
        self.content.append(f"<h{level}>{text}</h{level}>")
        
    def add_text(self, text: str):
        """Add a paragraph of text to the HTML file."""
        self.content.append(f"<p>{text}</p>")
        
    def add_image(self, image: np.ndarray, alt_text: str = "Image", 
                  width: Optional[int] = None, height: Optional[int] = None):
        """Add an image to the HTML file from a numpy array.
        
        Args:
            image: Numpy array of the image (uint8)
            alt_text: Alternative text for the image
            width: Optional width to display the image
            height: Optional height to display the image
        """
        if image.dtype != np.uint8:
            raise ValueError("Image must be of dtype uint8")
            
        # Convert numpy array to base64 encoded image
        img = Image.fromarray(image)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
        
        # Create style attribute if width or height is specified
        style = ""
        if width or height:
            style_parts = []
            if width:
                style_parts.append(f"width: {width}px")
            if height:
                style_parts.append(f"height: {height}px")
            style = f' style="{"; ".join(style_parts)}"'
            
        self.content.append(
            f'<div class="image-container">'
            f'<img src="data:image/png;base64,{img_str}" alt="{alt_text}"{style}>'
            f'</div>'
        )
    
    def add_raw_html(self, html: str):
        """Add raw HTML content directly."""
        self.content.append(html)
    
    def save(self, filename: str):
        """Save the HTML file to disk.
        
        Args:
            filename: Path to save the HTML file
        """
        with open(filename, 'w') as f:
            f.write(self._header)
            f.write('\n'.join(self.content))
            f.write(self._footer)
            
    def get_html(self) -> str:
        """Return the complete HTML as a string."""
        return self._header + '\n'.join(self.content) + self._footer