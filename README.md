# CyVidia
# Create a new virtual environment
python -m venv .myenvlocal

# Activate the virtual environment
source .myenvlocal/bin/activate   # On macOS/Linux
source activate .myenvlocal



# or
.\.myenvlocal\Scripts\activate   # On Windows

# Install the requirments 
python install requirments.txt


# Download and install the language model
python -m spacy download en_core_web_sm

# Now, run your script again
python file_name



# to install all dependencies 
pip install -r requirements.txt
