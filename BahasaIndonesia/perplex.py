import math
import kenlm

# load language model 
m = kenlm.Model('lm_pengadilan/6gram.bin')

# read corpus file 
file = open("newspapers.txt", "r")
s = file.read()

# count the perplexity 
list(m.full_scores(s))
n = len(s.split())
sum_inv_logs = -1 * sum(score for score, _, _ in m.full_scores(s))

# print(math.pow(sum_inv_logs, 1.0/n)) # for base 1 
print(math.pow(10.0, sum_inv_logs / n)) # for base 10 
