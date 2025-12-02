"""
Example script demonstrating how to use the distance calculator for video analysis.
"""

from distance_calculator import DistanceCalculator, VideoDistanceProcessor


def example_basic_usage():
    """
    Basic example: Calculate distances for known object widths.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Distance Calculation")
    print("=" * 70)
    
    # Initialize calculator with typical coastal bird dimensions
    # For example, a seagull is approximately 0.4m wide (40cm) with wings partially extended
    known_width = 0.4  # metres
    focal_length = 800  # pixels (calibrated value)
    
    calculator = DistanceCalculator(known_width, focal_length)
    
    print(f"\nSetup:")
    print(f"  Known bird width: {known_width}m")
    print(f"  Camera focal length: {focal_length}px")
    
    # Calculate distances for different scenarios
    print("\nDistance calculations for different perceived widths:")
    perceived_widths = [50, 100, 150, 200, 250, 300]
    
    for width_px in perceived_widths:
        distance = calculator.calculate_distance(width_px)
        print(f"  Bird appears {width_px}px wide -> Distance: {distance:.2f}m")
    
    print()


def example_calibration():
    """
    Example: Calibrate the camera focal length.
    """
    print("=" * 70)
    print("EXAMPLE 2: Camera Calibration")
    print("=" * 70)
    
    # Known bird width
    known_width = 0.35  # metres (35cm - medium-sized coastal bird)
    
    # Initial focal length estimate
    calculator = DistanceCalculator(known_width, focal_length=100)
    
    print(f"\nCalibrating camera for birds with {known_width}m width...")
    
    # Calibration scenario: Bird is known to be at 10 meters distance
    # and appears 280 pixels wide in the frame
    known_distance = 10.0  # metres
    perceived_width_at_distance = 280  # pixels
    
    print(f"\nCalibration data:")
    print(f"  Known distance: {known_distance}m")
    print(f"  Perceived width at that distance: {perceived_width_at_distance}px")
    
    # Calibrate
    focal_length = calculator.calibrate_focal_length(
        known_distance, 
        perceived_width_at_distance
    )
    
    print(f"\nCalibrated focal length: {focal_length:.2f}px")
    
    # Verify calibration
    print("\nVerification:")
    test_distance = calculator.calculate_distance(perceived_width_at_distance)
    print(f"  Using calibrated values, {perceived_width_at_distance}px -> {test_distance:.2f}m")
    print(f"  Expected: {known_distance}m")
    print(f"  Calibration {'successful' if abs(test_distance - known_distance) < 0.1 else 'needs adjustment'}!")
    
    print()


def example_video_processing():
    """
    Example: Process a video file (requires actual video file).
    """
    print("=" * 70)
    print("EXAMPLE 3: Video Processing")
    print("=" * 70)
    
    print("\nThis example shows how to process a video file.")
    print("Note: Requires an actual video file to process.\n")
    
    # Setup
    known_width = 0.3  # metres
    focal_length = 750  # pixels
    
    calculator = DistanceCalculator(known_width, focal_length)
    processor = VideoDistanceProcessor(calculator)
    
    print("Code example:")
    print("-" * 70)
    print("""
    # Initialize calculator and processor
    calculator = DistanceCalculator(known_width=0.3, focal_length=750)
    processor = VideoDistanceProcessor(calculator)
    
    # Process video
    distances = processor.process_video(
        video_path='path/to/video.mp4',
        output_path='output_video.mp4',  # Optional: save annotated video
        display=True  # Show video while processing
    )
    
    # Analyze results
    if distances:
        avg_distance = sum(d[1] for d in distances) / len(distances)
        print(f"Average distance: {avg_distance:.2f}m")
        
        min_distance = min(d[1] for d in distances)
        max_distance = max(d[1] for d in distances)
        print(f"Distance range: {min_distance:.2f}m - {max_distance:.2f}m")
    """)
    print("-" * 70)
    
    print("\nTo process your own video:")
    print("  1. Ensure you have a video file")
    print("  2. Calibrate the focal length for your camera")
    print("  3. Set the known width for your target species")
    print("  4. Run the processor on your video file")
    
    print()


def example_species_specific():
    """
    Example: Different configurations for various coastal bird species.
    """
    print("=" * 70)
    print("EXAMPLE 4: Species-Specific Configurations")
    print("=" * 70)
    
    # Different bird species have different average widths
    species_data = {
        'Seagull': {'width': 0.4, 'focal_length': 800},
        'Pelican': {'width': 0.6, 'focal_length': 800},
        'Tern': {'width': 0.25, 'focal_length': 800},
        'Albatross': {'width': 0.8, 'focal_length': 800},
        'Cormorant': {'width': 0.35, 'focal_length': 800},
    }
    
    print("\nPre-configured settings for common coastal birds:")
    print()
    
    for species, config in species_data.items():
        calculator = DistanceCalculator(
            known_width=config['width'],
            focal_length=config['focal_length']
        )
        
        print(f"{species}:")
        print(f"  Average width: {config['width']}m")
        
        # Example: bird appears 200px wide
        test_width = 200
        distance = calculator.calculate_distance(test_width)
        print(f"  If appears {test_width}px wide -> Distance: {distance:.2f}m")
        print()


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("DISTANCE CALCULATOR - USAGE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates various use cases for the distance calculator.")
    print()
    
    # Run examples
    example_basic_usage()
    example_calibration()
    example_video_processing()
    example_species_specific()
    
    print("=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nFor more information, see distance_calculator.py")
    print()


if __name__ == "__main__":
    main()
