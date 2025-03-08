#!/bin/bash

# Define the new version and commit/release messages
new_version="v0.1.13" # Replace with the actual version number
commit_message="Bump to $new_version" # Replace with the actual commit message
release_message="Release $new_version" # Replace with the actual release message

# Update version in pyproject.toml
echo "Updating version in pyproject.toml to $new_version..."
sed -i '' "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml
if [ $? -ne 0 ]; then
    echo "Error updating version in pyproject.toml"
    exit 1
fi

# Update version in __init__.py
echo "Updating version in __init__.py to $new_version..."
sed -i '' "s/__version__ = \".*\"/__version__ = \"${new_version#v}\"/" pyaging/__init__.py
if [ $? -ne 0 ]; then
    echo "Error updating version in __init__.py"
    exit 1
fi

# Run ruff for linting
echo "Running ruff for linting..."
ruff check pyaging --fix

# Run black for code formatting
echo "Running ruff for code formatting..."
ruff format pyaging
if [ $? -ne 0 ]; then
    echo "ruff formatting failed"
    exit 1
fi

# Run poetry update
echo "Running poetry update..."
poetry update
if [ $? -ne 0 ]; then
    echo "Poetry update failed"
    exit 1
fi

# Build the package
echo "Building the package..."
poetry build
if [ $? -ne 0 ]; then
    echo "Poetry build failed"
    exit 1
fi

# Install the package
echo "Installing the package..."
poetry install
if [ $? -ne 0 ]; then
    echo "Poetry install failed"
    exit 1
fi

# Update clocks and notebooks in the 'clocks/notebooks' directory
#echo "Updating clocks and notebooks..."
#cd clocks/notebooks
#total=$(ls *.ipynb | wc -l)
#counter=1
#for notebook in *.ipynb; do
#    # Skip the file if it is 'template.ipynb'
#    if [ "$notebook" = "template.ipynb" ]; then
#        echo "Skipping template.ipynb"
#        continue
#    fi
#
#    echo "Processing clock notebook ($counter/$total): $notebook"
#    #jupyter nbconvert --to notebook --execute "$notebook" #Change 
#    jupyter nbconvert --execute --inplace "$notebook" # Execute in place
#    if [ $? -ne 0 ]; then
#        echo "Error processing $notebook"
#        exit 1
#    fi
#    let counter=counter+1
#done
#cd ../..

# Run the script to update all clocks
#echo "Running script to update all clocks..."
#cd clocks
#python3 update_all_clocks.py $new_version
#if [ $? -ne 0 ]; then
#    echo "Updating clocks failed"
#    exit 1
#fi
#cd ..
#echo "Reminder: Upload all clocks and metadata to S3!"

# Process tutorials
#echo "Processing tutorials..."
#cd tutorials
#for notebook in *.ipynb; do
#    echo "Processing tutorial notebook: $notebook"
#    jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute --inplace "$notebook"
#done
#cd ..

# Run gold standard tests
echo "Running gold standard tests..."
poetry run pytest
if [ $? -ne 0 ]; then
    echo "Gold standard tests failed"
    exit 1
fi

# Run tutorial tests
echo "Running tutorial tests..."
poetry run pytest --nbmake tutorials/
if [ $? -ne 0 ]; then
    echo "Tutorial tests failed"
    exit 1
fi

# Build documentation
echo "Building documentation..."
cp tutorials/*.ipynb docs/source/tutorials
cp clocks/notebooks/*.ipynb docs/source/clock_notebooks
cd docs
make html
if [ $? -ne 0 ]; then
    echo "Documentation build failed"
    exit 1
fi
cd ..

# Commit and push changes
echo "Committing and pushing changes..."
git add .
git commit -m "$commit_message"
if [ $? -ne 0 ]; then
    echo "Git commit failed"
    exit 1
fi

git push
if [ $? -ne 0 ]; then
    echo "Git push failed"
    exit 1
fi

# Create and push tag
echo "Creating and pushing tag $new_version..."
git tag -a "$new_version" -m "$release_message"
git push origin "$new_version"
if [ $? -ne 0 ]; then
    echo "Git tag creation or push failed"
    exit 1
fi

echo "Version update pipeline completed successfully."
