import RNA
target = "((((....).))).((((....).))).((((....).))).((((....).))).((((....).))).((((....).))).((((((((((((((((((((((...)))).))))))).(((((..((((((((((..(((((((...))))))).))))))))))((((.((((((...)))))))))))))))...))))))))))).((((....).))).((((....).))).((((....).))).((((....).))).((((....).)))"
s="UUACGAAGGAUAAACCGCAGAUGACGGAAGGCGAAAGACCUACCACGAUAGAUGGAGCACGCAGGUUGCAGGGAAGCGUACCCGCUUCCUGCCAAUGUCCGCGUCUCCCGGGCUGUGGACAGGGUACGUUGAGCGUGUUCAUGCACGAUGGUCGUGCGCAACACGCUCGGGUGAUACACGGUGCGUGUACGCUGUGCCAACUUGGCAGGAAGAGUCCGUGAGAGACAGGCCGCAAGAGCCAAACCGAAGGAGUUAGCACGAAGGAUGCAGUAUUGACAAUAC"
print("len",len(s))
st,en=RNA.fold_compound(s).mfe()
print("bpd",RNA.bp_distance(st,target))
print("fold",st)
