#!/usr/bin/env python3
"""
XML to CSV Converter for Chinese vocabulary cards
Converts XML deck format to CSV with columns: zh, eng, score
"""

import xml.etree.ElementTree as ET
import csv
import argparse
import sys
from pathlib import Path


def parse_xml_to_csv(input_file, output_file):
    """
    Parse XML file and convert to CSV format
    
    Args:
        input_file (str): Path to input XML file
        output_file (str): Path to output CSV file
    """
    try:
        # Parse the XML file
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # Open CSV file for writing
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['zh', 'eng', 'score'])
            
            # Find all card elements
            cards = root.findall('.//card')
            
            if not cards:
                print("Warning: No cards found in the XML file")
                return
            
            # Process each card
            for card in cards:
                # Extract Chinese text
                chinese_elem = card.find('chinese[@name="Chinese"]')
                zh_text = chinese_elem.text if chinese_elem is not None else ''
                
                # Extract English meaning
                meaning_elem = card.find('text[@name="Meaning"]')
                eng_text = meaning_elem.text if meaning_elem is not None else ''
                
                # Set default score (you can modify this logic as needed)
                score = 0  # Default score, can be customized
                
                # Write row to CSV
                writer.writerow([zh_text, eng_text, score])
            
            print(f"Successfully converted {len(cards)} cards from {input_file} to {output_file}")
            
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when writing to '{output_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert XML vocabulary deck to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.xml output.csv
  %(prog)s -i deck.xml -o vocabulary.csv
        """
    )
    
    # Add arguments
    parser.add_argument(
        'input_file', 
        nargs='?',
        help='Input XML file path'
    )
    parser.add_argument(
        'output_file', 
        nargs='?',
        help='Output CSV file path'
    )
    parser.add_argument(
        '-i', '--input',
        dest='input_file_alt',
        help='Input XML file path (alternative)'
    )
    parser.add_argument(
        '-o', '--output',
        dest='output_file_alt',
        help='Output CSV file path (alternative)'
    )
    
    args = parser.parse_args()
    
    # Determine input and output files
    input_file = args.input_file or args.input_file_alt
    output_file = args.output_file or args.output_file_alt
    
    # Validate arguments
    if not input_file:
        parser.error("Input file is required")
    if not output_file:
        parser.error("Output file is required")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    # Convert XML to CSV
    parse_xml_to_csv(input_file, output_file)


if __name__ == '__main__':
    main()