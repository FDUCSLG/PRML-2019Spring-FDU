from model.LSTMsingle import *
from data.preposess import *
#===================================================================
# Dictionary
# map the char into a float list which is its vec
# store the words with ptrtrain
# for all the chars in my dict
print('Processing')
dict=build_pretrain_dict()
dictionary,dictionary_rev=build_dict()
D_dict=len(dictionary)
eight_lines,eight_lines_dev=clr_poem(dictionary,dictionary_rev)
num_of_poems=len(eight_lines)
num_of_poems_dev=len(eight_lines_dev)
print(num_of_poems)
print("The Dictionary has been built!")
#===================================================================
# Parameters
D_batch=32
D_input=300
D_H=128
T=40

char_each_poem=len(eight_lines[0])

def train_single():
    record_file=[]
    poemproducer=PoemProducerSingle(D_input,D_H,D_dict,dictionary,dict) 
    try:
        poemproducer.load_state_dict(torch.load('.record/single record loss largeset'))
        print('Load sucess!')
    except:
        print("No record yet. Build a new model and start!")
    
    lossfunc=CrossEntropy()
    optimizer = torch.optim.RMSprop(poemproducer.parameters())
    num_of_poems=len(eight_lines)
    poems_in_data=poemproducer.turn_char_poems_to_one_hots(eight_lines,dictionary)
    record=poemproducer.LSTM.W.sum().item()
    for t in range(T): # for each poem, train turns times
        Sum=0
        print("Epoch: T=%d" % (t))
        i=0
        torch.save(poemproducer.state_dict(),'.record/single record loss largeset')
        order=list(range(num_of_poems))
        random.shuffle(order)
        
        while i+D_batch<num_of_poems: 
            poems=torch.zeros(D_batch,char_each_poem,D_dict)
            for j in range(0,D_batch): # pick up some poems
                poems[j]=poems_in_data[order[(i+j)%num_of_poems]]
            
            h_last=torch.zeros(D_batch,poemproducer.D_H)
            c_last=torch.zeros(D_batch,poemproducer.D_H)
            LossSum=torch.zeros(1)
            optimizer.zero_grad()
            for char_ct in range(char_each_poem-1):
                
                # Get the predicted vectors
                pred_vec, h_last, c_last   = poemproducer(poems,char_ct,h_last,c_last)

                # Get the target vectors
                target_vec = poemproducer.get_targ_vec(poems,char_ct)
                
                loss=lossfunc(pred_vec,target_vec,D_dict)
                
                # loss.backward(retain_graph=True)
                LossSum+=loss
            LossSum.backward()
            optimizer.step()
            
            Sum+=LossSum
            print('LossSum: %.4f'%(LossSum/(D_batch*char_each_poem)))   
            i+=D_batch
        produce('夜',poemproducer,char_each_poem,dictionary,dictionary_rev)
        record_file.append((Sum/(num_of_poems*char_each_poem)).item()) 
        print('AvgLoss of this epoch: %0.6f'%(Sum/(num_of_poems*char_each_poem)))
    write('.record/loss/record single loss mrs largeset.txt',record_file)

def train_double():
    record_file=[]
    poemproducer=PoemProducerDouble(D_input,D_H,D_dict,dictionary,dict) 
    try:
        poemproducer.load_state_dict(torch.load('.record/double record'))
        print('Load sucess!')
    except:
        print("No record yet. Build a new model and start!")
    
    lossfunc=CrossEntropy()
    optimizer = torch.optim.RMSprop(poemproducer.parameters())
    num_of_poems=len(eight_lines)
    poems_in_data=poemproducer.turn_char_poems_to_one_hots(eight_lines,dictionary)
    poems_in_data_dev=poemproducer.turn_char_poems_to_one_hots(eight_lines_dev,dictionary)

    record=poemproducer.LSTM1.W.sum().item()
    for t in range(T): # for each poem, train turns times
        Sum=0
        Sum_dev=0
        Avg_dev=0
        Sum_tra=0

        print("Epoch: T=%d" % (t))
        i=0
        torch.save(poemproducer.state_dict(),'.record/double record')
        order=list(range(num_of_poems))
        random.shuffle(order)

        while i+D_batch<num_of_poems: 
            poems=torch.zeros(D_batch,char_each_poem,D_dict)
            for j in range(0,D_batch): # pick up some poems
                poems[j]=poems_in_data[order[(i+j)%num_of_poems]]
            
            h_last1=torch.zeros(D_batch,poemproducer.D_H)
            c_last1=torch.zeros(D_batch,poemproducer.D_H)
            h_last2=torch.zeros(D_batch,poemproducer.D_H)
            c_last2=torch.zeros(D_batch,poemproducer.D_H)
            LossSum_tra=torch.zeros(D_batch)

            LossSum=torch.zeros(1)
            # LossPPL=torch.zeros(D_batch)
            optimizer.zero_grad()
            for char_ct in range(char_each_poem-1):
                
                # Get the predicted vectors
                pred_vec, h_last1, c_last1, h_last2, c_last2   = poemproducer(poems,char_ct,h_last1,c_last1,h_last2,c_last2)

                # Get the target vectors
                target_vec = poemproducer.get_targ_vec(poems,char_ct)
                
                # loss_ppl=cross_entropy(pred_vec,target_vec)
                
                LossSum+=lossfunc(pred_vec,target_vec,D_dict)
                
                LossSum_tra+=cross_entropy(pred_vec,target_vec,D_dict)
                # LossPPL+=loss_ppl
            LossSum_tra=torch.exp(LossSum_tra/char_each_poem).sum()
            Sum_tra+=LossSum_tra
            LossSum.backward()
            optimizer.step()
            
            Sum+=LossSum
            # LossPPL=torch.exp(LossPPL/char_each_poem).sum()/D_batch
            print('LossAvg: %.4f'%(LossSum/(D_batch*char_each_poem)))  
            i+=D_batch
        
        # Dev set control
        # if 1:
        #     i=0
        #     order_dev=list(range(num_of_poems_dev))
        #     random.shuffle(order_dev)
        #     while i+D_batch<num_of_poems_dev: 
        #         poems_dev=torch.zeros(D_batch,char_each_poem,D_dict)            
        #         for j in range(D_batch): # pick up some poems
        #             poems_dev[j]=poems_in_data_dev[order_dev[(i+j)%num_of_poems_dev]]

        #         h_last1_dev=torch.zeros(D_batch,poemproducer.D_H)
        #         c_last1_dev=torch.zeros(D_batch,poemproducer.D_H)
        #         h_last2_dev=torch.zeros(D_batch,poemproducer.D_H)
        #         c_last2_dev=torch.zeros(D_batch,poemproducer.D_H)

        #         LossSum_dev=torch.zeros(D_batch)

        #         for char_ct in range(char_each_poem-1):

        #             # Get the predicted vectors
        #             pred_vec_dev, h_last1_dev, c_last1_dev, h_last2_dev, c_last2_dev   = poemproducer(poems_dev,char_ct,h_last1_dev,c_last1_dev,h_last2_dev,c_last2_dev)

        #             # Get the target vectors
        #             target_vec_dev = poemproducer.get_targ_vec(poems_dev,char_ct)

        #             loss_dev=cross_entropy(pred_vec_dev,target_vec_dev,D_dict)

        #             LossSum_dev+=loss_dev
        #         LossSum_dev=torch.exp(LossSum_dev/char_each_poem).sum()

        #         i+=D_batch
        #         Sum_dev+=LossSum_dev
            # Sum_dev/=num_of_poems_dev
        record_file.append((Sum/(num_of_poems*char_each_poem)).item()) 
        produce('红',poemproducer,char_each_poem,dictionary,dictionary_rev) 
        print('Sum of this epoch: %0.6f'%(Sum/(num_of_poems*char_each_poem)))
    write('.record/record double loss largeset.txt',record_file)

def train_triple():
    poemproducer=PoemProducerTriple(D_input,D_H,D_dict,dictionary,dict) 
    try:
        poemproducer.load_state_dict(torch.load('.record/triple record'))
        print('Load sucess!')
    except:
        print("No record yet. Build a new model and start!")
    
    lossfunc=CrossEntropy()
    optimizer = torch.optim.Adam(poemproducer.parameters(), lr=0.001)
    num_of_poems=len(eight_lines)
    poems_in_data=poemproducer.turn_char_poems_to_one_hots(eight_lines,dictionary)
    record=poemproducer.LSTM1.W.sum().item()
    for t in range(T): # for each poem, train turns times
        Sum=0
        print("Epoch: T=%d" % (t))
        i=0
        torch.save(poemproducer.state_dict(),'.record/triple record')
        order=list(range(num_of_poems))
        random.shuffle(order)
        
        while i+D_batch<num_of_poems: 
            poems=torch.zeros(D_batch,char_each_poem,D_dict)
            for j in range(0,D_batch): # pick up some poems
                poems[j]=poems_in_data[order[(i+j)%num_of_poems]]
            
            h_last1=torch.zeros(D_batch,poemproducer.D_H)
            c_last1=torch.zeros(D_batch,poemproducer.D_H)
            
            h_last2=torch.zeros(D_batch,poemproducer.D_H)
            c_last2=torch.zeros(D_batch,poemproducer.D_H)

            h_last3=torch.zeros(D_batch,poemproducer.D_H)
            c_last3=torch.zeros(D_batch,poemproducer.D_H)


            LossSum=torch.zeros(1)
            optimizer.zero_grad()
            for char_ct in range(char_each_poem-1):
                
                # Get the predicted vectors
                pred_vec, h_last1, c_last1, h_last2, c_last2, h_last3, c_last3   = poemproducer(poems,char_ct,h_last1,c_last1,h_last2, c_last2, h_last3, c_last3)

                # Get the target vectors
                target_vec = poemproducer.get_targ_vec(poems,char_ct)
                
                loss=lossfunc(pred_vec,target_vec,D_dict)
                
                # loss.backward(retain_graph=True)
                LossSum+=loss
            LossSum.backward()
            optimizer.step()
            
            Sum+=LossSum
            print('Change: %.4f  LossSum: %.4f'%(abs(poemproducer.LSTM1.W.sum().item()-record),LossSum) )  
            i+=D_batch
        produce('日',poemproducer,char_each_poem,dictionary,dictionary_rev) 
        print('Sum of this epoch: %0.6f'%Sum)

def main():
    while 1:
        print('Input the model you want to look at...')
        print('Input 1 for single version,')
        print('2 for double version...')
        print('3 for double version...')
        print('4 for get a poem by single version')
        print('5 for get a poem by doubel version')
        print('6 for get a poem by triple version')
        print('7 to quit')
        model=input('Input something:  ')
        if model=='1':
            train_single()
        elif model=='2':
            train_double()
        elif model=='3':
            train_triple()
        elif model=='4':
            poemproducer=PoemProducerSingle(D_input,D_H,D_dict,dictionary,dict) 
            poemproducer.load_state_dict(torch.load('.record/single record'))
            char=input('Input a char to begin')
            poem=produce(char,poemproducer,50,dictionary,dictionary_rev)
            print(poem)
        elif model=='5':
            poemproducer=PoemProducerDouble(D_input,D_H,D_dict,dictionary,dict) 
            poemproducer.load_state_dict(torch.load('.record/double record'))
            char=input('Input a char to begin')
            poem=produce(char,poemproducer,50,dictionary,dictionary_rev)
            print(poem)
        elif model=='6':
            poemproducer=PoemProducerTriple(D_input,D_H,D_dict,dictionary,dict) 
            poemproducer.load_state_dict(torch.load('.record/triple record'))
            char=input('Input a char to begin')
            poem=produce(char,poemproducer,50,dictionary,dictionary_rev)
            print(poem)
        elif model=='7':
            break
        else:
            print('Try another time')
            continue
    print('Thanks for using the system!')

main()