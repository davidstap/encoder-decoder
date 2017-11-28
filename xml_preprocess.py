from __future__ import with_statement
from lxml import etree
import sys
from nltk.tokenize import sent_tokenize, word_tokenize

fn = sys.argv[1]
output = sys.argv[2]

node=etree.parse('data/'+fn)
sentences = node.xpath("//mteval//doc//seg/text()")

cleaned = []
for s in sentences:
	cleaned.append([word_tokenize(s.replace("'", " ").replace('"',"")) for s in sent_tokenize(s.strip())][0])

with open(output, 'w') as f:
    for _list in cleaned:
         f.write(' '.join(_list) + '\n')