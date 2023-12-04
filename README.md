# CyVidia
# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate   # On macOS/Linux
source activate .venv



# or
.\.venv\Scripts\activate   # On Windows

# Install the requirments 
pip install -r requirements.txt 


# Download and install the language model
python -m spacy download en_core_web_sm

# Now, run your script again
python file_name



# to install all dependencies 
pip install -r requirements.txt