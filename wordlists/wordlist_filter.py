import json
import os

def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst

def filter_repeat_word(wordlist):
    final_wordlist = []
    for i in range(len(wordlist)):
        if wordlist[i] not in final_wordlist:
            final_wordlist.append(wordlist[i])
    return final_wordlist

if __name__ == '__main__':
    ster_wordlist = load_word_list("stereotype-selected.txt")
    print("length of stereotype wordlist before filter:{}".format(len(ster_wordlist)))

    filtered_ster_wordlist = filter_repeat_word(ster_wordlist)
    print("length of stereotype wordlist after filter:{}".format(len(filtered_ster_wordlist)))

    write_file = "filtered_stereotype.txt"
    f = open(write_file,'w')
    for phrase in filtered_ster_wordlist:
        f.write(phrase)
        f.write("\n")
    f.close()