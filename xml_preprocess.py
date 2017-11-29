from __future__ import with_statement
from lxml import etree
import sys
from nltk.tokenize import sent_tokenize, word_tokenize

fn = sys.argv[1]
output = sys.argv[2]

# node=etree.parse('data/'+fn)
# sentences = node.xpath("//mteval//doc//seg/text()")

# cleaned = []
# for s in sentences:
# 	cleaned.append([word_tokenize(s.replace("'", " ").replace('"',"")) for s in sent_tokenize(s.strip())][0])

# with open('data_cleaned/'+output, 'w') as f:
#     for _list in cleaned:
#          f.write(' '.join(_list).encode('utf-8') + '\n')


node = etree.parse('data/'+fn)

sentences = node.xpath("//doc//text()")
sentences = [s for s in sentences if len(s) > 2000]

for line in sentences[0].splitlines():
	print(line)
	cleaned_line = [word_tokenize(s) for s in line.split()]
	if len(cleaned_line) > 0:
		print(cleaned_line)
# 	print([word_tokenize(s) for s in sent_tokenize(line)])

#print([word_tokenize(s) for s in sent_tokenize('Thank you very much, I appreciate it.')])

#print(sentences[0])

	