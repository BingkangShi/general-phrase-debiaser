# General-Phrase-Debiaser
code and model files of paper"General Phrase Debiaser: Debiasing Masked Language Models at a Multi-Token Level"

## Download model parameters
You can download the debiased model parameters through Google Drive.

The downloaded model file contains three folders, corresponding to three different models: Bert, Albert and DistilBert. Please put these three folders under the "model" path.

## Dependency
Before running any python file, please ensure that your environment meets the following dependencies:

python >= 3.6

torch >= 1.6.0

transformers >= 4.26

dataset >= 2.6.0

evaluate >= 0.4.0

matplotlib >= 3.6.2

And make sure your GPU has no less than 24G of memory

## Start evaluation
Take bert-base-uncased as an example:

1.Generate biased prompts through running:

python generate_prompts.py --debias_type gender --model_type bert

and then you can get prompt file from the path data/prompts_bert-base_gender

2.Debias by fine-tuning the model:

python auto-debias.py --debias_type gender --model_name_or_path bert-base-uncased --model_type bert --prompts_file data/prompts_bert-base_gender --epochs 5

The fine-tuned model will be stored under the path "model".

3.Evaluation:

After debias the model, you can execute the following command to perform the SEAT test:

python SEAT-eval/experiments/sent_eval.py --model_path data/prompts_bert-base_gender --model_type bert --output_name prompts_bert-base_gender --results_dir SEAT-eval/experiments/result/debiased-bert

You can also use the GLUE test to evaluate the language ability of the debiased model. The terminal command of GLUE test is in GLUE_test/terminal.txt .
