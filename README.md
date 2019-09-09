# Auto TL;DR (A Text Summarizer)
====
## Description
As its name suggests, Auto TL;DR automatically generates summaries for news articles. This program is built with Python, using spaCy.py as a natural language processing toolkit.

## Running Instructions
1. A list of stop words, in the format of a python file named stop_list.py, must be placed in the same directory of the source code. This file is the same one provided in the NLP class.

2. The input text file must be placed in the same directory of the source code. The paragraphs of the input file must be separated by two new lines (\n\n). (See the provided sample input, nafta.txt, as an example)

3. The following python packages are required in order to run the code:

spacy
numpy
nltk
sklearn
scipy

Per spaCy's documentation, a larger version of the spaCy package needs to be downloaded in order for the word vectorization functionality to work. Said package can be downloaded by executing the following line in Terminal:

python -m spacy download en_core_web_lg

4. To run the code:
python auto-tldr.py [input file name]

For example:
python auto-tldr.py nafta.txt

5. The output files will be in this format:
output[number].txt

For example:
output3.txt, output4.txt, etc.

The number indicates the number of paragraphs in the summary.
