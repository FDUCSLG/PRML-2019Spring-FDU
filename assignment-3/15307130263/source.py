import torch
import fastNLP
from fastNLP import DataSet
from fastNLP import Trainer
from fastNLP import Batch
from fastNLP import CrossEntropyLoss
from fastNLP import core
from fastNLP.core.optimizer import Adam
from fastNLP.core.optimizer import SGD
import data_process
import model

if __name__ == '__main__':
    tpds = data_process.TangPoemDataset(maxLength=20,useBigData=True,useSmallData=False)
    tpds.loadCharEmbedding()
    m = model.HCLSTM(numEmbd=tpds.totalWords,hidden_size=300,weight=tpds.weight,embedding=tpds.embedding,usePreEmbedding=True)
    m.cuda()
    ADAMOP = Adam()
    SGDMmOp = SGD(lr=0.001,momentum=0.9)
    trainner = Trainer(
        tpds.trainSet,
        model = m,
        check_code_level=0,
        n_epochs=150,
        batch_size=128,
        metric_key="PPL",
        dev_data=tpds.testSet,
        metrics=core.metrics.AccuracyMetric(target="output_s"),
        optimizer=ADAMOP
    )
    trainner.train()
    torch.save(m.state_dict(),'model.pkl')
    #m.load_state_dict(torch.load('model.pkl'))
    m = m.cpu()
    pred = m.runStartWith('日',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
    pred = m.runStartWith('红',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
    pred = m.runStartWith('山',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
    pred = m.runStartWith('夜',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
    pred = m.runStartWith('湖',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
    pred = m.runStartWith('海',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
    pred = m.runStartWith('月',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab))
'''
if __name__ == '__main__':
    tpds = data_process.TangPoemDataset(maxLength=20,useBigData=True,useSmallData=False)
    tpds.loadCharEmbedding()
    m = model.HCLSTM(numEmbd=tpds.totalWords,hidden_size=300,weight=tpds.weight,embedding=tpds.embedding,usePreEmbedding=True)
    m.cuda()
    m.load_state_dict(torch.load('model.pkl'))
    m = m.cpu()
    pred = m.runStartWith('日',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
    pred = m.runStartWith('红',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
    pred = m.runStartWith('山',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
    pred = m.runStartWith('夜',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
    pred = m.runStartWith('湖',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
    pred = m.runStartWith('海',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
    pred = m.runStartWith('月',tpds.vocab,19)
    print(m.convertOutput(pred,tpds.vocab,k=1))
'''
# with weight top10 model1
#  日酴荏柄规,靮妓晁坌缅,焦嚣陷窜釜,洼廉队绨淬
#  红藕挈邸犯,完澶尉纳邯,厨陛览驳隽,铢泸缩曷衲
#  山棚禹朽窣,板鹄浇缣苇,品堠肆噞鲍,秾谿唳鸑隳
#  夜臼缱桤醁,汩婵浣聿遨,栊蹙叩遏弊,榜准嫭寮庚
#  湖涨飧盆袯,邳蜜魁忙删,菀桠坡均嗢,羌皛厄葵憩
#  海峤式针铃,陨眸槱鲈犀,徂衲偎茗鷕,滂授突诘级
#  月瓣滉辕浆,勒述财诣鹳,赫焯毡郤鹧,研矫拌绅侬
# without weight top10 model2
#  日 暖千兢综,饷魍沓班额,曀肮赁贬騧,块爟葵跻兑
#  红 妆貔燋沙,跪匕荠啐喔,蠡缉弩陀糅,嗤刳廛铗嗛
#  山 光灶呴亏,簳炳货壬蜜,苎黴酲赩鄂,熊灿蔑捶鹔
#  夜 色吟勾夹,面苫郭汇镏,鳅允瑜灺旁,陡瀼燮抆跌
#  湖 南绾攘彼,颐蕣湍岷蟋,炫蟢褊颖悒,羿蜜眩遮诧
#  海 上舞夜蚕,首煦瞩鲭禋,瀰澡奁疟驴,艟舣铩屹惶
#  月 落山兢官,刁确箕泌踯,呙孰荏龉产,鶗彘喉奠洮
# without weight top1 model2
#  日 暮吟微摈,噤娜偓鼯它,滹謏瞒晏滂,楠爟椅魍熨
#  红 颜炽队楼,救婀吁阀姹,袂遒赌闼砑,榉鲧允弁骛
#  山 川貔庵梓,辎炳豸醑爨,刍滇酲滇褭,祔泮祯蓑摈
#  夜 雨郎丝委,拨摐蘅伎晃,黤賨澶愠姻,蠢忒缮朅镏
#  湖 上禾攘彼,硗滉湍巩弗,炫娃騧颖笃,鍮挟眩苫丐
#  海 上呈尽域,薅礴恺鲭熙,胧澡奁谧倖,跗忏妄瞪惶
#  月 明乘祸闽,蜂膂嬉诺岫,咳帙咳晶皙,括怵哉溯刬