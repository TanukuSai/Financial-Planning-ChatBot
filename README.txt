This is a webapp that uses IBM Granite. The Model is hosted on the computer that will be running the program so ensure your computer is up to the task or use google colab to run it.
To run the program on a windows pc, make sure the file "main.py" , "requirements.txt" , "run.bat" are in the same folder and then simply double click the run.bat file to execute.
You might see a huge wall of text but don't panic that just ensures that the dependencies and packages required to run the chatbot are installed and if they aren't it installs them for you.
If you run the app on collab add the following lines at the start of the code:
!pip install -q transformers accelerate bitsandbytes gradio pandas openpyxl pydantic matplotlib
!pip install -q gTTS
