"""
SVG Generator
Converts handwriting strokes to SVG vector paths
"""
import svgwrite
import numpy as np
from typing import List, Tuple


def strokes_to_paths(strokes, scale=1.0, offset=(0, 0)):
    """
    Convert stroke sequence to SVG path commands
    
    Args:
        strokes: [N, 3] array of (dx, dy, pen_state)
                pen_state: 0 = pen down (drawing), 1 = pen up (move)
        scale: Scaling factor for stroke coordinates
        offset: (x, y) offset for positioning
    
    Returns:
        paths: List of path strings (one per continuous stroke)
    """
    if len(strokes) == 0:
        return []
    
    paths = []
    current_path = []
    
    # Convert cumulative displacements to absolute positions
    x, y = offset[0], offset[1]
    positions = [(x, y)]
    
    for dx, dy, pen_state in strokes:
        x += dx * scale
        y += dy * scale
        positions.append((x, y))
    
    # Build path commands
    current_path = []
    for i, (x, y) in enumerate(positions[1:]):  # Skip first point
        pen_state = strokes[i][2]
        
        if len(current_path) == 0:
            # Start new path
            current_path.append(f"M {x:.2f},{y:.2f}")
        else:
            # Continue path
            current_path.append(f"L {x:.2f},{y:.2f}")
        
        # If pen up or last point, finalize this path
        if pen_state > 0.5 or i == len(strokes) - 1:
            if len(current_path) > 0:
                paths.append(" ".join(current_path))
                current_path = []
    
    # Add any remaining path
    if len(current_path) > 0:
        paths.append(" ".join(current_path))
    
    return paths


def smooth_strokes(strokes, window_size=3):
    """
    Apply smoothing to strokes using moving average
    
    Args:
        strokes: [N, 3] array of strokes
        window_size: Size of smoothing window
    
    Returns:
        smoothed: Smoothed strokes
    """
    if len(strokes) < window_size:
        return strokes
    
    smoothed = np.copy(strokes)
    half_window = window_size // 2
    
    for i in range(half_window, len(strokes) - half_window):
        # Smooth x and y coordinates
        smoothed[i, 0] = np.mean(strokes[i-half_window:i+half_window+1, 0])
        smoothed[i, 1] = np.mean(strokes[i-half_window:i+half_window+1, 1])
        # Keep pen state unchanged
    
    return smoothed


def create_svg(strokes_list, output_path, width=800, height=1000,
               stroke_width=2, stroke_color='black', smooth=True,
               scale=1.0, margin=50):
    """
    Create an SVG file from handwriting strokes
    
    Args:
        strokes_list: List of stroke arrays, each [N, 3] (for multiple lines/words)
        output_path: Path to save SVG file
        width: SVG canvas width
        height: SVG canvas height
        stroke_width: Width of stroke lines
        stroke_color: Color of strokes
        smooth: Whether to apply smoothing
        scale: Scaling factor for strokes
        margin: Margin around content
    
    Returns:
        None (saves file to output_path)
    """
    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')
    
    # Add white background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    
    # Current vertical position for text lines
    current_y = margin
    line_spacing = 60  # Space between lines
    
    for strokes in strokes_list:
        if len(strokes) == 0:
            continue
        
        # Convert to numpy if needed
        if isinstance(strokes, list):
            strokes = np.array(strokes)
        
        # Apply smoothing if requested
        if smooth:
            strokes = smooth_strokes(strokes, window_size=3)
        
        # Calculate bounding box for this line
        cumsum_x = np.cumsum(strokes[:, 0])
        cumsum_y = np.cumsum(strokes[:, 1])
        
        # Center the line horizontally
        min_x = np.min(cumsum_x)
        max_x = np.max(cumsum_x)
        line_width = (max_x - min_x) * scale
        offset_x = margin + (width - 2 * margin - line_width) / 2
        
        # Convert strokes to SVG paths
        paths = strokes_to_paths(strokes, scale=scale, offset=(offset_x, current_y))
        
        # Add paths to SVG
        for path_d in paths:
            path = dwg.path(d=path_d,
                          stroke=stroke_color,
                          stroke_width=stroke_width,
                          fill='none',
                          stroke_linecap='round',
                          stroke_linejoin='round')
            dwg.add(path)
        
        # Move to next line
        line_height = (np.max(cumsum_y) - np.min(cumsum_y)) * scale
        current_y += max(line_height, 30) + line_spacing
    
    # Save SVG
    dwg.save()
    print(f"SVG saved to {output_path}")


def create_full_page_svg(text_lines, model, style_images, char_to_idx,
                        output_path, device='cpu', **svg_kwargs):
    """
    Generate a full page of handwriting from multiple lines of text
    
    Args:
        text_lines: List of text strings (one per line)
        model: Trained handwriting synthesis model
        style_images: Style reference images [1, N, 1, H, W]
        char_to_idx: Character to index mapping
        output_path: Path to save SVG
        device: Device to run model on
        **svg_kwargs: Additional arguments for create_svg
    
    Returns:
        None (saves SVG file)
    """
    import torch
    from ..models.content_encoder import text_to_indices
    
    model.eval()
    all_strokes = []
    
    with torch.no_grad():
        for line_text in text_lines:
            if not line_text.strip():
                # Empty line - add spacing
                all_strokes.append(np.array([[0, 0, 1]]))  # Just a placeholder
                continue
            
            # Convert text to indices
            indices, mask = text_to_indices(line_text, char_to_idx, max_len=None)
            
            # Prepare tensors
            char_indices = torch.tensor([indices], dtype=torch.long, device=device)
            char_mask = torch.tensor([mask], dtype=torch.float, device=device)
            style_imgs = style_images.to(device)
            
            # Generate strokes
            strokes = model.generate(char_indices, style_imgs, char_mask,
                                   points_per_char=10, temperature=0.7)
            
            # Convert to numpy
            strokes_np = strokes[0].cpu().numpy()
            all_strokes.append(strokes_np)
    
    # Create SVG
    create_svg(all_strokes, output_path, **svg_kwargs)


def batch_generate_pages(texts_list, model, style_images, char_to_idx,
                         output_dir, device='cpu', **svg_kwargs):
    """
    Generate multiple pages of handwriting
    
    Args:
        texts_list: List of text lists (each inner list is one page)
        model: Trained model
        style_images: Style references
        char_to_idx: Character mapping
        output_dir: Directory to save SVG files
        device: Device to run on
        **svg_kwargs: SVG generation arguments
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, text_lines in enumerate(texts_list):
        output_path = os.path.join(output_dir, f'page_{i+1:03d}.svg')
        create_full_page_svg(text_lines, model, style_images, char_to_idx,
                           output_path, device, **svg_kwargs)
        print(f"Generated page {i+1}/{len(texts_list)}")


def svg_to_pdf(svg_path, pdf_path):
    """
    Convert SVG to PDF (requires cairosvg)
    
    Args:
        svg_path: Input SVG file path
        pdf_path: Output PDF file path
    """
    try:
        import cairosvg
        cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
        print(f"PDF saved to {pdf_path}")
    except ImportError:
        print("cairosvg not installed. Install it to enable PDF export.")
        print("pip install cairosvg")
