import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs
import sys


def cut_words(sentence):
    # print sentence
    return " ".join(jieba.cut(sentence)).encode('utf-8')


inp, outp = sys.argv[1:3]
# f = codecs.open('wiki.zh.jian.text', 'r', encoding="utf8")
# target = codecs.open("zh.jian.wiki.seg-1.3g.txt", 'w', encoding="utf8")
f = codecs.open(inp, 'r', encoding="utf8")
target = codecs.open(outp, 'w', encoding="utf8")
print('open files')
line_num = 1
line = f.readline()
while line:
    if line_num % 1000 == 0:
        print('---- processing ', line_num, ' article----------------')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()
print('---- processing ', line_num, ' article----------------')
f.close()
target.close()
exit()
while line:
    curr = []
    for oneline in line:
        # print(oneline)
        curr.append(oneline)
    after_cut = map(cut_words, curr)
    target.writelines(after_cut)
    print('saved', line_num, 'articles')
    exit()
    line = f.readline1()
f.close()
target.close()

# python Testjieba.py
