from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import matplotlib.pylab as plt


def load_file(fn):
	with open(fn) as f:
		lines = f.readlines()
	return lines

def pretty_print(references, hypotheses):
	for i in range(len(hypotheses)):
		print(' '.join(references[i][0]))
		print(' '.join(hypotheses[i]))
		print(sentence_bleu([references[i][0]], hypotheses[i]))

		print('*'*30)

def find_length_n(references, hypotheses, n):
    ref_list = []
    hyp_list = []
    
    for i in range(len(hypotheses)):
    	if len(references[i]) == n:
    		ref_list.append(references[i])
    		hyp_list.append(hypotheses[i])
    		# [r for r in ref_list] OR [[r] for r in ref_list] ???

    return ref_list, hyp_list, corpus_bleu([r for r in ref_list], hyp_list)

references = [ref.split() for ref in load_file('nl.txt')]
hypotheses = [hyp.split() for hyp in load_file('preds.txt')]


#pretty_print(references, hypotheses)

# hypothesis == prediction

#print('BLEU SCORE: ',corpus_bleu([[r] for r in references], hypotheses))

scores = []
bleu_scores = dict()
for i in range(2,42):
	print(i)

	ref_len, hyp_len, bleu = find_length_n(references, hypotheses, i)

	for j in range(len(ref_len)):
		print('ref:\t ', ' '.join(ref_len[j]))
		print('hyp:\t ', ' '.join(hyp_len[j]))
		print(sentence_bleu([ref_len[j]], hyp_len[j]))
		print('-'*20)


	print('BLEU score for length '+str(i)+': '+str(bleu))
	print('*'*80)
	bleu_scores[i] = bleu



#pretty_print([[r] for r in ref_len2], hyp_len2)
#print('bleu2', bleu2)


# questions: 
# # [r for r in ref_list] OR [[r] for r in ref_list] ??? when calculating BLEU score


lists = sorted(bleu_scores.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.show()



# prediction in space: is context of sentence better measure than BLEU?



