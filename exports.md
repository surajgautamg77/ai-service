#!/bin/bash

# --- Configuration ---
OUTPUT_FILE="whole_project_structure.md"

# 1. Exclude specific directories/files ONLY from the project root
ROOT_EXCLUDE_ARRAY=( 
    ".git" 
    ".vscode" 
    "$OUTPUT_FILE" # Always exclude the output file itself
    "export.md"
    "venv"
    "package-lock.json"
)

# 2. NEW: Exclude any directory that matches these names, ANYWHERE in the project
PATTERN_EXCLUDE_ARRAY=(
    "node_modules"
    "dist"
    "build"
    "coverage"
    "__pycache__" # Example for Python projects
)

# 3. Exclude files with these extensions from being read
BINARY_EXTENSIONS=( "png" "jpg" "jpeg" "gif" "ico" "svg" "webp" "woff" "woff2" "ttf" "eot" "otf" "pdf" "zip" "gz" "tar" "rar" "exe" "dll" "so" "a" "lib" "jar" "mp3")

# --- Dynamically build exclusion patterns ---

# For 'tree', both root and pattern exclusions work the same way
COMBINED_TREE_EXCLUDES=("${ROOT_EXCLUDE_ARRAY[@]}" "${PATTERN_EXCLUDE_ARRAY[@]}")
TREE_EXCLUDE_PATTERN=""
for item in "${COMBINED_TREE_EXCLUDES[@]}"; do
    TREE_EXCLUDE_PATTERN+="$item|"
done
TREE_EXCLUDE_PATTERN=${TREE_EXCLUDE_PATTERN%|}

# For 'find', we build the two types of rules separately
ROOT_FIND_ARGS=()
for item in "${ROOT_EXCLUDE_ARRAY[@]}"; do
    ROOT_FIND_ARGS+=(-o -path "./$item")
done

PATTERN_FIND_ARGS=()
for item in "${PATTERN_EXCLUDE_ARRAY[@]}"; do
    # Use -name to match the directory name anywhere
    PATTERN_FIND_ARGS+=(-o -name "$item")
done

# Combine all directory exclusion rules for 'find'
COMBINED_FIND_ARGS=("${ROOT_FIND_ARGS[@]}" "${PATTERN_FIND_ARGS[@]}")
COMBINED_FIND_ARGS=("${COMBINED_FIND_ARGS[@]:1}") # Remove the initial '-o'

# Build binary exclusion rules as before
BINARY_EXCLUDE_ARGS=()
for ext in "${BINARY_EXTENSIONS[@]}"; do
    BINARY_EXCLUDE_ARGS+=(-o -iname "*.$ext")
done
BINARY_EXCLUDE_ARGS=("${BINARY_EXCLUDE_ARGS[@]:1}")

# --- Script ---

# 1. Create the file header and the clean directory tree
{
    echo "# Project Structure"
    echo ""
    echo "\`\`\`"
    # Use the native Linux 'tree' command with the --prune flag for clarity
    tree -aF --prune -I "$TREE_EXCLUDE_PATTERN"
    echo "\`\`\`"
    echo ""
    echo "# File Contents"
} > "$OUTPUT_FILE"

# 2. Find, filter, and append file contents
# Use the new combined rules to prune directories, then filter out binaries
find . \( "${COMBINED_FIND_ARGS[@]}" \) -prune -o -type f -not \( "${BINARY_EXCLUDE_ARGS[@]}" \) -print | while IFS= read -r file; do
    relativePath=$(echo "$file" | sed 's|^\./||')
    extension="${relativePath##*.}"
    if [[ "$relativePath" == "$extension" ]]; then extension="text"; fi
    
    # Append (>>) the content for each file to the existing file
    {
        echo "---"
        echo "File: $relativePath"
        echo "---"
        echo ""
        echo "\`\`\`$extension"
        # Use 'sed' to remove Windows carriage return ('\r') characters for LLM compatibility
        sed 's/\r$//' "$file"
        echo ""
        echo "\`\`\`"
        echo ""
    } >> "$OUTPUT_FILE"
done

echo "✅ Project successfully exported to '$OUTPUT_FILE' (patterns and binaries ignored)."