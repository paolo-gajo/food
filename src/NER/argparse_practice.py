import argparse

def process_file(filename, length, verbose):
    """
    Dummy file processing function.
    """
    print(f"Processing file: {filename}")
    if verbose:
        print(f"Processing length: {length}")
        print("Verbose mode is on.")

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add arguments
    parser.add_argument('filename', type=str, help='The name of the file to process.')
    parser.add_argument('--length', type=int, default=10, help='Length of the process.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')

    # Parse arguments
    args = parser.parse_args()

    # Pass the arguments to the file processing function
    process_file(args.filename, args.length, args.verbose)

if __name__ == '__main__':
    main()
