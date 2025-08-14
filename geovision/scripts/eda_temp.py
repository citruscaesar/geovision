class MultiFileReader:
    """Allows reading from multiple files sequentially as a single stream."""
    def __init__(self, filenames):
        self.filenames = filenames
        self.file_iter = iter(filenames)
        self.current_file = open(next(self.file_iter), "rb")
        print(f"Opened first archive part: {self.filenames[0]}")

    def read(self, size=-1):
        """Read bytes from the current file, switching to the next file when needed."""
        data = self.current_file.read(size)
        while not data and self.file_iter:
            self.current_file.close()
            try:
                next_file = next(self.file_iter)
                print(f"Switching to next archive part: {next_file}")
                self.current_file = open(next_file, "rb")
                data = self.current_file.read(size)
            except StopIteration:
                break
        return data

    def close(self):
        """Close the last opened file."""
        print("Closing final file.")
        self.current_file.close()

def list_tar_gz_files(part_files, output_file="file_list.txt"):
    """
    Lists the contents of a split .tar.gz archive without loading the entire archive into memory.

    :param part_files: List of tar.gz parts in correct order.
    :param output_file: Path to save the file listing.
    """
    print(f"Starting to process {len(part_files)} archive parts...")
    
    # Open the output file
    with open(output_file, "w") as out_file:
        print(f"Writing file list to {output_file}")
        
        # Open multiple parts as a concatenated stream
        print("Opening tar.gz parts as a continuous stream...")
        with tarfile.open(fileobj=gzip.GzipFile(fileobj=MultiFileReader(part_files)), mode="r|") as tar:
            for member in tar:
                # Write file names to output file
                out_file.write(member.name + "\n")
                print(f"Found file: {member.name}")  # Also print to console for real-time feedback
    
    print("File listing completed.")

class MultiFileReader:
    """Reads multiple files sequentially as a single stream while ensuring valid tar alignment."""
    def __init__(self, filenames):
        self.filenames = filenames
        self.file_iter = iter(self.filenames)
        self.current_file = None
        self._open_next_file()

    def _open_next_file(self):
        """Open the next available file in sequence."""
        if self.current_file:
            self.current_file.close()
        try:
            next_file = next(self.file_iter)
            print(f"Opening archive part: {next_file}")
            self.current_file = open(next_file, "rb")
        except StopIteration:
            self.current_file = None

    def read(self, size=-1):
        """Read bytes from the current file, switching to the next file when needed."""
        if not self.current_file:
            return b""  # No more data
        
        data = self.current_file.read(size)
        if not data:  # End of current file, move to next
            self._open_next_file()
            return self.read(size)  # Recursive call to continue reading
        return data

    def close(self):
        """Close the last opened file."""
        if self.current_file:
            print("Closing final file.")
            self.current_file.close()

def list_tar_files_from_idx(part_files, output_file="file_list.txt", start_index=0):
    """
    Lists the contents of a split .tar archive without loading the entire archive into memory,
    ensuring alignment with a valid tar header.

    :param part_files: List of tar parts in correct order.
    :param output_file: Path to save the file listing.
    :param start_index: The index of the first file to include in the listing (0-based).
    """
    if start_index >= len(part_files):
        print("Error: Start index is out of range.")
        return
    
    print(f"Processing archives starting from index {start_index} ({part_files[start_index]} onwards)...")
    
    # Open the output file
    with open(output_file, "w") as out_file:
        print(f"Writing file list to {output_file}")
        
        # Open all parts as a continuous stream, ensuring we start at a valid tar header
        reader = MultiFileReader(part_files)
        with tarfile.open(fileobj=reader, mode="r|*") as tar:
            skipped = 0
            for member in tar:
                if skipped < start_index:
                    skipped += 1
                    continue  # Skip until we reach the desired start index
                out_file.write(member.name + "\n")
                #print(f"Found file: {member.name}")  # Log output
    
    print("File listing completed.")
