# Encoder-Decoder model with attention

## Ideas
- Create other (better?) measure than BLEU: compute e.g. Euclidean distance of context vectors.
- See if this new measure is similar to what BLEU finds


## TO DO
- Find out how model deals with encoder-decoder: there is only one .pt file?
- Find out exactly what data has been used for training (description on Blackboard has changed)
- Tidy up 'bleu.py' (make more general, accept arguments, etc.)

## Open questions
- Bleu score: should references be in a list? Plot makes more sense when references are not in a list.
- Is the idea good? Would this be sufficient for project?
