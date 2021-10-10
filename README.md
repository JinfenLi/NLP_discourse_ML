# A RST Discourse Parser with new nuclearity type of N~ and new action of R~

We build up a RST discourse parser described in [tree Representations in Transition System for RST Parsing] based on the paper of http://aclanthology.coli.uni-saarland.de/pdf/P/P17/P17-2029.pdf. 

Due to the licence of RST data corpus, we can't include the data in our project folder. To reproduce the result in the paper, you need to download it from the LDC. Put the TRAINING and TEST folders under data directory

### Usage:
1. Prepare Virtual Environment

    >conda create -n rst_env python=3.6
                                   
    >conda activate rst_env
                                   
    >pip install requirements.txt
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
<!-- 2. Download Resources:

    
    This project relies on Stanford CoreNLP toolkit to preprocess the data. You can download from [here](http://stanfordnlp.github.io/CoreNLP/index.html) and put the file [run_corenlp.sh](./run_corenlp.sh) into the CoreNLP folder. Then use the following command to preprocess both the data in train_dir and in test_dir. -->
    
2. Configurations

    >directory of data: --data_dir [../data/TRAINING | ../data/TEST]
                      
    >directory of model or any output: --output_dir saved_model
                      
    >directory of Stanford CoreNLP tool: --corenlp_dir [your stanford nlp path]
                      
    >parse type, including Treebank and any other type of data: --parse_type [Treebank | Your data with the dir name under folder of data]
                      
    >Set True to use RN\~model and False to use N\~model: --isFlat [True|False]
                      
    >use Stanford CoreNLP tool to parse data first: --preprocess [True|False]
                                                                                                                                                                                                                                                                                                                                                                                                             
    >whether to extract feature templates, action maps and relation maps: --prepare [True|False]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    >whether to train on Treebank: --train [True|False]

    >whether to evaluate on Treebank: --eval [True|False]

    >whether to predict any data: --pred [True|False]
### Requirements:

All the codes are tested under Python 3.6. And see requirements.txt for python library dependency and you can install them with pip.


