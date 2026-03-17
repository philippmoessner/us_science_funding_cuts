"""
UI utility functions for the OpenAlex Mapper Gradio app.
"""

from openalex_utils import openalex_url_to_readable_name


def highlight_queries(text: str) -> str:
    """Split OpenAlex URLs on semicolons and display them as colored pills with readable names."""
    palette = ["#f5f5f5", #set to  only light grey
        # "#e8f4fd", "#fff2e8", "#f0f9e8", "#fdf2f8",
        # "#f3e8ff", "#e8f8f5", "#fef7e8", "#f8f0e8"
    ]
    
    # Handle empty input
    if not text or not text.strip():
        return "<div style='padding: 10px; color: #666; font-style: italic;'>Enter OpenAlex URLs separated by semicolons to see query descriptions</div>"
    
    # Split URLs on semicolons and strip whitespace
    urls = [url.strip() for url in text.split(";") if url.strip()]
    
    if not urls:
        return "<div style='padding: 10px; color: #666; font-style: italic;'>No valid URLs found</div>"
    
    pills = []
    for i, url in enumerate(urls):
        color = palette[i % len(palette)]
        try:
            # Get readable name for the URL
            readable_name = openalex_url_to_readable_name(url)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            readable_name = f"Query {i+1}"
        
        pills.append(
            f'<span style="background:{color};'
            'padding: 8px 12px; margin: 4px; '
            'border-radius: 12px; font-weight: 500;'
            'display: inline-block; font-family: \'Roboto Condensed\', sans-serif;'
            'border: 1px solid rgba(0,0,0,0.1); font-size: 14px;'
            'box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'
            f'{readable_name}</span>'
        )
    
    return (
        "<div style='padding: 8px 0;'>"
        "<div style='font-size: 12px; color: #666; margin-bottom: 6px; font-weight: 500;'>"
        f"{'Query' if len(urls) == 1 else 'Queries'} ({len(urls)}):</div>"
        "<div style='display: flex; flex-wrap: wrap; gap: 4px;'>"
        + "".join(pills) + 
        "</div></div>"
    ) 