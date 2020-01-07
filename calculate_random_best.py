from pytt.utils import read_pickle

counts_file = '/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes/counts.pkl'
rebalanced = True
threshold = 50

counts = read_pickle(counts_file)

micro_counts = [[], [], []]
macro_scores = [[], [], []]

for k,v in counts.items():
    if v[0] < threshold or v[1] < threshold: continue
    total = v[0]+v[1]
    true_positives = v[1]/2 if rebalanced else v[1]*v[1]/total
    micro_counts[0] += [true_positives]
    positives = total/2 if rebalanced else v[1]
    micro_counts[1] += [positives]
    relevants = v[1]
    micro_counts[2] += [relevants]
    if positives != 0:
        p = true_positives/positives
        macro_scores[0] += [p]
    if relevants != 0:
        r = true_positives/relevants
        macro_scores[1] += [r]
    if positives != 0 or relevants != 0:
        f1 = 2*p*r/(p+r)
        macro_scores[2] += [f1]

# macro
p, r, f1 = sum(macro_scores[0])/len(macro_scores[0]), sum(macro_scores[1])/len(macro_scores[1]), sum(macro_scores[2])/len(macro_scores[2])
print("macro_averaged: p - %f, r - %f, f1 - %f" % (p, r, f1))

# micro
p, r = sum(micro_counts[0])/sum(micro_counts[1]), sum(micro_counts[0])/sum(micro_counts[2])
f1 = 2*p*r/(p+r)
print("micro_averaged: p - %f, r - %f, f1 - %f" % (p, r, f1))
