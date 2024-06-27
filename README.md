# Evaluating the Cognitive Plausibility of LSTM and Transformer Language Models Across Different Languages

## Introduction
This repository contains all the code used for the thesis titled “Evaluating the Cognitive Plausibility of LSTM and Transformer Language Models Across Distinct Linguistic Structures”. This thesis explores the challenge of mimicking human language comprehension in natural language processing. We compare the cognitive plausibility of two models, an LSTM model and a Transformer model with masked self-attention, focusing on their performance in Hindi and English, two languages with distinct syntactic structures. By analyzing surprisal and entropy, this work evaluates how these models simulate human reading behavior, revealing insights into the cognitive processes during language processing.

### Repository Structure
•	Data File: Includes training data and sentences from eye-tracking datasets MECO and Potsdam, along with the Splitted data file which contains the training, testing and validation datasets to ensure the same datasets are used for both models. 

•	Language Models File: Contains scripts to train and test both LSTM and Transformer models.

•	LM Results File: Stores output from the trained language models, detailing performance metrics and analysis.

•	Merge Datasets File: Code to integrate eye-tracking data metrics (like first pass reading time) with model outputs (entropy and surprisal values) from the eye-tracking sentences.

•	Preprocessing File: Scripts for preparing the training data, ensuring it's formatted and ready for model training.

•	R Analysis File: Includes R studio code for statistical analysis, correlating eye-tracking data with model outputs and documenting these findings.
