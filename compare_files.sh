#!/bin/bash

# Define the directories
dir1="hazards"
dir2="hazardsOld"

# Check if both directories exist
if [ ! -d "$dir1" ]; then
    echo "Directory '$dir1' does not exist!"
    exit 1
fi

if [ ! -d "$dir2" ]; then
    echo "Directory '$dir2' does not exist!"
    exit 1
fi

# Loop through all files in the first directory
for file1 in "$dir1"/*; do
    # Check if it's a file (not a subdirectory)
    if [ -f "$file1" ]; then
        # Extract the file name from the full path
        filename=$(basename "$file1")
        file2="$dir2/$filename"

        # Check if the corresponding file exists in the second directory
        if [ -f "$file2" ]; then
            # Compare the two files
            if cmp -s "$file1" "$file2"; then
                echo "Files '$filename' are identical."
            else
                echo "Files '$filename' differ."
            fi
        else
            echo "File '$filename' does not exist in '$dir2'."
        fi
    fi
done
