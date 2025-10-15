"""
IAM-OnDB Data Converter
Converts IAM On-Line Handwriting Database to our model format

Usage:
    python convert_iam.py --iam_path /path/to/IAM-OnDB --output processed_iam.json
"""
import xml.etree.ElementTree as ET
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_iam_xml(xml_path):
    """
    Parse IAM-OnDB XML file
    
    Args:
        xml_path: Path to XML file
    
    Returns:
        List of strokes as (x, y, timestamp) tuples
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    strokes = []
    
    # Find all stroke elements
    for stroke in root.findall('.//Stroke'):
        stroke_points = []
        
        # Get all points in this stroke
        for point in stroke.findall('Point'):
            x = float(point.get('x'))
            y = float(point.get('y'))
            # Handle both int and float timestamps - convert float to int
            time_str = point.get('time', '0')
            time = int(float(time_str)) if time_str else 0
            stroke_points.append((x, y, time))

        if stroke_points:  # Only add non-empty strokes
            strokes.append(stroke_points)

    return strokes


def convert_to_relative(strokes):
    """
    Convert absolute coordinates to relative displacements

    Args:
        strokes: List of stroke lists, each containing (x, y, time) tuples

    Returns:
        List of [dx, dy, pen_state] triplets
    """
    result = []
    last_x, last_y = 0, 0

    for stroke_idx, stroke in enumerate(strokes):
        for point_idx, (x, y, time) in enumerate(stroke):
            dx = x - last_x
            dy = y - last_y

            # Pen is down (0) during stroke, up (1) at end of stroke
            pen_state = 0
            if point_idx == len(stroke) - 1 and stroke_idx < len(strokes) - 1:
                pen_state = 1  # Pen up at end of stroke (except last)

            result.append([float(dx), float(dy), float(pen_state)])
            last_x, last_y = x, y

    return result


def load_txt_file(txt_path):
    """
    Load text from a single .txt file

    Args:
        txt_path: Path to .txt file (e.g., a01-000u.txt)

    Returns:
        List of text lines
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except Exception as e:
        print(f"Warning: Could not read {txt_path}: {e}")
        return []


def get_transcription_for_xml(xml_path, ascii_root):
    """
    Get transcription for a specific XML file

    Args:
        xml_path: Path to XML file (e.g., a01-000u-01.xml)
        ascii_root: Root path to ascii directory

    Returns:
        Transcription text or empty string if not found
    """
    # Parse filename: a01-000u-01.xml
    filename = xml_path.stem  # a01-000u-01
    parts = filename.split('-')

    if len(parts) < 3:
        return ''

    participant = parts[0]  # a01
    sample = parts[1]       # 000u
    line_num = parts[2]     # 01

    # Build path to txt file: ascii-all/ascii/a01/a01-000/a01-000u.txt
    txt_filename = f"{participant}-{sample}.txt"
    txt_path = ascii_root / participant / f"{participant}-{sample[:3]}" / txt_filename

    if not txt_path.exists():
        return ''

    # Read all lines from txt file
    text_lines = load_txt_file(txt_path)

    # Get the specific line (line numbers are 1-indexed in filenames)
    try:
        line_index = int(line_num) - 1
        if 0 <= line_index < len(text_lines):
            return text_lines[line_index]
    except ValueError:
        pass

    return ''


def extract_writer_id(file_path):
    """
    Extract writer ID from IAM file path
    e.g., 'a01-000u-01.xml' -> writer 'a01' -> numeric ID

    Args:
        file_path: Path to XML file

    Returns:
        Integer writer ID
    """
    filename = os.path.basename(file_path)
    writer_code = filename.split('-')[0]

    # Convert writer code to numeric ID
    # a01 -> 1, a02 -> 2, b01 -> 27, etc.
    if len(writer_code) >= 3:
        letter = ord(writer_code[0]) - ord('a')
        number = int(writer_code[1:])
        writer_id = letter * 100 + number
        return writer_id

    return 0


def convert_iam_dataset(iam_path, output_path, max_samples=None, min_strokes=5):
    """
    Convert entire IAM-OnDB dataset to our format

    Args:
        iam_path: Root path to IAM-OnDB
        output_path: Output JSON file path
        max_samples: Maximum number of samples (None = all)
        min_strokes: Minimum number of strokes per sample

    Returns:
        Number of samples processed
    """
    print("Converting IAM-OnDB to model format...")

    # Find paths based on new structure
    iam_root = Path(iam_path)

    # Try different possible structures
    possible_stroke_paths = [
        iam_root / 'lineStrokes-all' / 'lineStrokes',
        iam_root / 'lineStrokes',
    ]

    possible_ascii_paths = [
        iam_root / 'ascii-all' / 'ascii',
        iam_root / 'ascii',
    ]

    stroke_path = None
    for path in possible_stroke_paths:
        if path.exists():
            stroke_path = path
            break

    ascii_path = None
    for path in possible_ascii_paths:
        if path.exists():
            ascii_path = path
            break

    if not stroke_path:
        print(f"Error: Stroke path not found. Tried:")
        for path in possible_stroke_paths:
            print(f"  - {path}")
        return 0

    if not ascii_path:
        print(f"Warning: ASCII path not found. Tried:")
        for path in possible_ascii_paths:
            print(f"  - {path}")
        print("Will use filenames as text (not recommended)")

    print(f"Using stroke path: {stroke_path}")
    if ascii_path:
        print(f"Using ASCII path: {ascii_path}")

    # Find all XML files
    xml_files = list(stroke_path.rglob('*.xml'))
    print(f"Found {len(xml_files)} XML files")

    if max_samples:
        xml_files = xml_files[:max_samples]
        print(f"Processing first {max_samples} samples")

    # Convert each file
    processed_data = []
    skipped = 0
    missing_transcriptions = 0

    for xml_file in tqdm(xml_files, desc="Converting"):
        try:
            # Parse XML
            strokes = parse_iam_xml(str(xml_file))

            if len(strokes) < min_strokes:
                skipped += 1
                continue

            # Convert to relative coordinates
            relative_strokes = convert_to_relative(strokes)

            # Get transcription
            if ascii_path:
                text = get_transcription_for_xml(xml_file, ascii_path)
                if not text:
                    missing_transcriptions += 1
                    text = xml_file.stem  # Use filename as fallback
            else:
                text = xml_file.stem

            # Extract writer ID
            writer_id = extract_writer_id(str(xml_file))

            # Create sample
            sample = {
                'strokes': relative_strokes,
                'text': text,
                'writer_id': writer_id,
                'source_file': str(xml_file.relative_to(iam_root))
            }

            processed_data.append(sample)

        except Exception as e:
            print(f"\nError processing {xml_file}: {e}")
            skipped += 1

    # Save to JSON
    print(f"\nSaving {len(processed_data)} samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)

    print(f"✓ Conversion complete!")
    print(f"  - Processed: {len(processed_data)} samples")
    print(f"  - Skipped: {skipped} samples (min_strokes={min_strokes})")

    if missing_transcriptions > 0:
        print(f"  - Missing transcriptions: {missing_transcriptions} samples (used filenames instead)")

    if processed_data:
        print(f"  - Unique writers: {len(set(s['writer_id'] for s in processed_data))}")

        # Statistics
        total_strokes = sum(len(s['strokes']) for s in processed_data)
        avg_strokes = total_strokes / len(processed_data)
        print(f"  - Average strokes per sample: {avg_strokes:.1f}")

        # Check how many have real text vs filenames
        real_text = sum(1 for s in processed_data if not s['text'].endswith('.xml') and '-' not in s['text'][-10:])
        print(f"  - Samples with transcriptions: {real_text} ({real_text/len(processed_data)*100:.1f}%)")
    else:
        print(f"\n⚠ WARNING: No samples were successfully processed!")
        print(f"  This could be due to:")
        print(f"  - Incorrect IAM-OnDB path structure")
        print(f"  - min_strokes threshold too high (current: {min_strokes})")
        print(f"  - Corrupted or incompatible XML files")

    return len(processed_data)


def validate_converted_data(json_path):
    """
    Validate converted data

    Args:
        json_path: Path to JSON file

    Returns:
        True if valid, False otherwise
    """
    print(f"\nValidating {json_path}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("⚠ No data to validate - file is empty")
        return False

    issues = []

    for i, sample in enumerate(data[:100]):  # Check first 100
        # Check fields
        if 'strokes' not in sample:
            issues.append(f"Sample {i}: Missing strokes")
        if 'text' not in sample:
            issues.append(f"Sample {i}: Missing text")
        if 'writer_id' not in sample:
            issues.append(f"Sample {i}: Missing writer_id")

        # Check strokes format
        strokes = sample.get('strokes', [])
        for j, stroke in enumerate(strokes[:10]):  # Check first 10 strokes
            if len(stroke) != 3:
                issues.append(f"Sample {i}, stroke {j}: Wrong format (expected [dx, dy, pen])")
                break

    if issues:
        print("⚠ Validation issues found:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Validation passed!")
        print(f"  - Total samples: {len(data)}")
        print(f"  - Sample text lengths: min={min(len(s['text']) for s in data)}, max={max(len(s['text']) for s in data)}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Convert IAM-OnDB to model format')
    parser.add_argument('--iam_path', type=str, required=True,
                       help='Path to IAM-OnDB root directory')
    parser.add_argument('--output', type=str, default='iam_processed.json',
                       help='Output JSON file path')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--min_strokes', type=int, default=5,
                       help='Minimum number of strokes per sample')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output after conversion')
    
    args = parser.parse_args()
    
    # Convert
    num_samples = convert_iam_dataset(
        args.iam_path,
        args.output,
        max_samples=args.max_samples,
        min_strokes=args.min_strokes
    )
    
    if num_samples > 0 and args.validate:
        validate_converted_data(args.output)
    
    print(f"\nDone! Use this data with:")
    print(f"  python train.py --data_path {args.output}")


if __name__ == '__main__':
    main()