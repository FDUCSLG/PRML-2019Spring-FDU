import os
os.sys.path.append("..")
import handout
import numpy as np
import matplotlib.pyplot as plt


dataset_train, dataset_test = handout.get_text_classification_datasets()

class_list = dataset_train.target_names
training_labels = dataset_train.target
testing_labels = dataset_test.target

temp = class_list[0].split(".")
string = temp[0]+"."+temp[1]+"\n"+temp[2]+"."+temp[3]
class_list[0] = string

plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(class_list))

performance = [0,0,0,0]
performance2 = [0,0,0,0]

for i in training_labels:
    performance[i]+=1
for i in testing_labels:
    performance2[i]+=1



total_width, n = 0.8, 2
width = total_width / n
y_pos=y_pos - (total_width - width) / 2

b=ax.barh(y_pos, performance, align='center',
          color='green', ecolor='black',height=0.2,label='training set')

for rect in b:
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,'%d'%w,ha='left',va='center')

b=ax.barh(y_pos+width, performance2, align='center',
          color='red', ecolor='black',height=0.2,label='testing set')

for rect in b:
    w=rect.get_width()
    ax.text(w,rect.get_y()+rect.get_height()/2,'%d'%w,ha='left',va='center')

ax.set_yticks(y_pos+width/2.0)
ax.set_yticklabels(class_list, fontsize=13)
ax.invert_yaxis()
ax.set_xlabel('Counts')
ax.set_title('Data Set Statistics', fontsize=20)
plt.legend(loc='best', prop={'size':14})
plt.show()
