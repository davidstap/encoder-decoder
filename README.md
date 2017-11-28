# Encoder-Decoder model with attention

## Ideas
- Create other (better?) measure than BLEU: compute e.g. Euclidean distance or cosine similarity of context vectors.
- See if this new measure is similar to what BLEU finds

## TO DO
- Find out how model deals with encoder-decoder: there is only one .pt file?
- Generalize up 'bleu.py' (make more general, accept arguments, etc.): now 'hardcoded' for dev set.
- Calculate BLEU score on train set (due to unclear instructions I did it on dev set +- 1000 entries)
- Calculate BLEU score on test set (due to unclear instructions I did it on dev set +- 1000 entries)

## DONE
- Find out exactly what data has been used for training (description on Blackboard has changed)
- Calculate BLEU score on dev set

## Open questions
- Bleu score: should references be in a list? Plot makes more sense when references are not in a list.
- Is the idea good? Would this be sufficient for project?

## Project / code description
- Model is trained on `data/train.tags.en-nl.en` and `data/train.tags.en-nl.nl` (+- 25000 entries).
- Dev set can be found here: `IWSLT17.TED.dev2010.en-nl.en.xml` and `IWSLT17.TED.dev2010.en-nl.nl.xml`. (+- 1000 entries)
- Test set can be found here: `IWSLT17.TED.tst2017.mltlng.en-nl.en.xml` and `IWSLT17.TED.tst2017.mltlng.nl-en.nl.xml`. (+- 1250 entries) 


- When running the code make sure you are located in the root folder.

Command to preprocess the TED data, both English and Dutch:
```
python xml_preprocess.py IWSLT17.TED.dev2010.en-nl.en.xml en.txt
python xml_preprocess.py IWSLT17.TED.dev2010.en-nl.nl.xml nl.txt
```

Command to translate `input_data.txt` using trained model (.pt file), result is stored in `write_to.txt`
```
python OpenNMT-py/translate.py -model OpenNMT-py/trained_models/ted_sgd_acc_55.43_ppl_12.39_e11.pt -src en.txt -output preds.txt -replace_unk -verbose
```

Command to calculate BLEU score and show plot:
```
python bleuscore.py
```
(should be extended to accept arguments etc.)

Note that inside folder `OpenNMT-py` some folders can be ignored (these are for educational purposes): we don't use `data`, `test`, and some other files. 

In `trained_models` there are two models: the one starting with `ted` is the one we need.

I used `txtdata` to play around, this can be ignored.
