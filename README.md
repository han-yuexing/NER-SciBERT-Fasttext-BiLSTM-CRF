# NER-SciBERT-Fasttext-BiLSTM-CRF
This is a material text-oriented named entity recognition model. The SciBERT and Fasttext word vectors are fused and input into the BiLSTM-CRF network to extract material entities. Our paper is on the way.

## Project File Structure
* fasttext
    * fasttext_embeddings-MINIFIED.model
    * fasttext_embeddings-MINIFIED.model.vectors.npy
    * fasttext_embeddings-MINIFIED.model.vectors_ngrams.npy
    * fasttext_embeddings-MINIFIED.model.vectors_vocab.npy
    * ___Tip: You need to download these models yourself___
* package
    * dataset.py
    * metrics.py
    * model.py
    * nn.py
    * utils.py
* resource
    * scibert-hface
        * config.json
        * pytorch_model.bin
        * special_tokens_map.json
        * tokenizer.json
        * tokenizer_config.json
        * vocab.txt
        * ___Tip: You need to download these models yourself___
* sls
    * train.json
    * val.json
* bert_w2v_main.py

## How to use?
Clone the warehouse code, download `SciBERT` and `fasttext` models according to the cited references, put the language model file in the corresponding directory, and then run `main.py`.

## Dataset
To create our NER dataset, we collected 250 English-language stainless steel papers. We conducted processing on 250 scientific papers, from which we extracted 2,453 sentences, and applied sequence labeling to these sentences using the [Doccano](https://github.com/doccano/doccano).
### Entity category
1. Material Name
2. Research Aspect
3. Technology
4. Method
5. Property
6. Property Value
7. Experiment Name
8. Experiment Condition
9. Condition Value
10. Experiment Output
11. Equipment Used
12. Involved Element
13. Applicable Scenario

## Language model
* URL of the original paper of `SciBERT`: https://aclanthology.org/D19-1371
* URL of the original paper of `Fasttext`: https://doi.org/10.1021/acs.jcim.9b00995
